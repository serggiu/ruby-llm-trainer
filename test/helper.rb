# frozen_string_literal: true

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

at_exit do
  puts
  if $failures.zero?
    puts "OK — #{$assertions} assertions passed"
  else
    puts "#{$failures}/#{$assertions} assertions FAILED"
    exit 1
  end
end
