#!/usr/bin/env ruby
# frozen_string_literal: true

# Poll system-wide free memory (memory_pressure -Q) every 2 s.
#
# Usage: ruby tools/memmon.rb [duration_seconds]  (default 1800)
#
# Prints timestamped lines like:
#   [12:00:03] free: 18.4 GB (57.5%)
#
# Used to observe RAM during a training run: start it in the background,
# run training, then read the log.

POLL_SECONDS = 2

def total_mem_gb
  out = IO.popen(["sysctl", "-n", "hw.memsize"], &:read)
  out.strip.to_f / (1024.0**3)
end

def free_mem_gb(total_gb)
  out = IO.popen(["memory_pressure", "-Q"], &:read)
  m = out.match(/System-wide memory free percentage:\s*([\d.]+)%/)
  m ? total_gb * m[1].to_f / 100.0 : total_gb # unknown → assume fine
end

duration = (ARGV[0] || 1800).to_i
total = total_mem_gb

deadline = Time.now + duration
while Time.now < deadline
  free = free_mem_gb(total)
  puts format("[%s] free: %5.1f GB (%4.1f%%)",
              Time.now.strftime("%H:%M:%S"), free, free / total * 100)
  $stdout.flush
  sleep POLL_SECONDS
end
