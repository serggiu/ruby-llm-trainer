# frozen_string_literal: true

require "stringio"

# Minimal assertion helpers shared by test/test_*.rb. A failing assertion is
# counted and printed; the process exits non-zero if any assertion failed
# (checked via at_exit), so bin/test can rely on the exit status.

$failures = 0
$assertions = 0

def assert(cond, label = "assertion")
  $assertions += 1
  return if cond

  $failures += 1
  puts "  FAIL: #{label}"
end

def assert_equal(expected, actual, label = nil)
  assert(expected == actual, label || "expected #{expected.inspect}, got #{actual.inspect}")
end

def assert_nil(value, label = "expected nil, got #{value.inspect}")
  assert(value.nil?, label)
end

def assert_raises(error_class, label = "raises #{error_class}")
  $assertions += 1
  yield
  $failures += 1
  puts "  FAIL: #{label} — nothing raised"
  nil
rescue error_class => e
  # expected — return the exception so callers can inspect it
  e
end

# Runs the block with stdout/stderr silenced — for calls that intentionally
# produce warnings (e.g. tests of malformed-input handling).
def quietly
  orig_out, orig_err = $stdout, $stderr
  $stdout = $stderr = StringIO.new
  yield
ensure
  $stdout, $stderr = orig_out, orig_err
end

at_exit do
  puts
  if $failures.zero?
    puts "OK — #{$assertions} assertions passed"
  else
    puts "#{$failures}/#{$assertions} assertions FAILED"
    exit 1
  end
end
