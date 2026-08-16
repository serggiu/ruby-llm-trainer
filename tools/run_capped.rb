#!/usr/bin/env ruby
# frozen_string_literal: true

# Run a training command, optionally under a memory watchdog.
#
# macOS does not allow hard per-process memory limits (setrlimit for
# RLIMIT_AS/RLIMIT_DATA returns EINVAL), so a true "28 GB cap" is impossible
# at the kernel level. Instead, when enabled, this supervisor polls
# system-wide free memory and SIGKILLs the training process as soon as free
# memory drops below a threshold — the run aborts with an error instead of
# swapping the machine into a freeze.
#
# The watchdog is opt-in. By default the command is executed directly (this
# script is a transparent pass-through: signals and exit codes behave exactly
# as if the command had been run without it). Enable supervision per run with
# the WATCHDOG environment variable:
#
#   WATCHDOG=1 ruby tools/run_capped.rb [--min-free-gb 4] -- COMMAND [ARGS...]
#
# With a 32 GB machine and --min-free-gb 4, training effectively cannot use
# more than ~28 GB.
#
# Example:
#   WATCHDOG=1 ruby tools/run_capped.rb --min-free-gb 4 -- .venv/bin/mlx_lm.lora \
#       --model _mlx/model_ornith_4bit --train --data _mlx/cpt_data ...

POLL_SECONDS = 2
WATCHDOG_VALUES = %w[1 true yes on].freeze

def watchdog_enabled?
  WATCHDOG_VALUES.include?(ENV.fetch("WATCHDOG", "").strip.downcase)
end

def total_mem_gb
  out = IO.popen(["sysctl", "-n", "hw.memsize"], &:read)
  out.strip.to_f / (1024.0**3)
end

def free_mem_gb(total_gb)
  out = IO.popen(["memory_pressure", "-Q"], &:read)
  m = out.match(/System-wide memory free percentage:\s*([\d.]+)%/)
  m ? total_gb * m[1].to_f / 100.0 : total_gb # unknown → assume fine
end

# Parses ARGV: leading options (--min-free-gb N) until the first non-option,
# everything from there on (a leading "--" separator is stripped) is the
# command. Returns [command_array, min_free_gb].
def parse_args(argv)
  min_free_gb = 4.0

  until argv.empty?
    case argv.first
    when "--min-free-gb"
      argv.shift
      value = argv.shift
      begin
        min_free_gb = Float(value)
      rescue ArgumentError, TypeError
        warn "run_capped: --min-free-gb requires a number, got #{value.inspect}"
        exit 2
      end
    when /\A--min-free-gb=(.+)\z/
      begin
        min_free_gb = Float(Regexp.last_match(1))
      rescue ArgumentError
        warn "run_capped: --min-free-gb requires a number, got #{Regexp.last_match(1).inspect}"
        exit 2
      end
      argv.shift
    when "--"
      argv.shift
      break
    else
      break
    end
  end

  [argv, min_free_gb]
end

def kill_and_wait(pid)
  Process.kill("KILL", pid)
  Process.wait(pid)
rescue Errno::ESRCH, Errno::ECHILD
  # child already gone — nothing to do
end

def main
  cmd, min_free_gb = parse_args(ARGV.dup)

  if cmd.empty?
    warn "usage: #{$PROGRAM_NAME} [--min-free-gb N] -- COMMAND [ARGS...]"
    exit 2
  end

  unless watchdog_enabled?
    # Pass-through: replace this process with the command. Signals and exit
    # codes behave exactly as if it had been run directly.
    begin
      exec(*cmd)
    rescue SystemCallError => e
      warn "run_capped: failed to exec #{cmd[0]}: #{e.message}"
      exit 127
    end
  end

  total = total_mem_gb
  if total <= 0
    warn "run_capped: could not determine total memory (sysctl hw.memsize)"
    exit 2
  end
  puts format("[watchdog] %.0f GB total RAM, killing child below %.1f GB free", total, min_free_gb)

  pid = Process.spawn(*cmd)

  # Ctrl-C (SIGINT): kill the child and exit 130, like the interpreter's
  # KeyboardInterrupt handling in the original Python version.
  trap("INT") do
    kill_and_wait(pid)
    exit 130
  end

  loop do
    sleep POLL_SECONDS

    begin
      _, status = Process.waitpid2(pid, Process::WNOHANG)
      if status
        code = status.exitstatus || -status.termsig.to_i
        puts format("[watchdog] child exited with code %d", code)
        exit(code)
      end
    rescue Errno::ECHILD
      exit 0
    end

    free = free_mem_gb(total)
    next unless free < min_free_gb

    puts format(
      "[watchdog] FREE MEMORY CRITICAL: %.1f GB free (< %.1f GB) — killing child to prevent a system freeze",
      free, min_free_gb
    )
    kill_and_wait(pid)
    puts "[watchdog] child killed. Reduce --max-seq-length (or the dataset size) " \
         "so the run fits under the memory cap, then retry."
    exit 1
  end
end

main
