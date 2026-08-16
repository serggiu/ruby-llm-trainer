#!/usr/bin/env ruby
# frozen_string_literal: true

# Data root — where _sources/, _dataset/, etc. live. Override with
# LLM_TRAINER_ROOT (used by tests to run the pipeline in a sandbox).
ROOT = ENV["LLM_TRAINER_ROOT"] || File.expand_path("..", File.dirname(__FILE__))

# Generate the source-attribution file (Attribution.md) from the repositories
# under _sources/.
#
# For every repository it reads the license file found at the repo root
# (LICENSE*, MIT-LICENSE*, COPYING*, ...), detects the license type from the
# license text, extracts the copyright line, and gets the origin URL from the
# git remote. The result is written to Attribution.md at the project root.
#
# Attribution.md is git-ignored: it lists the exact repositories used for
# local training and is not meant to be committed (see .gitignore). It is
# regenerated whenever main.rb runs, so it always matches _sources/.
#
# Fully source-agnostic: it never names or hardcodes any repository — adding,
# removing, or renaming a repo under _sources/ and re-running updates the file
# automatically. A repo without a license file is skipped with a warning.
#
# Usage:
#   ruby build/build_attribution.rb [sources_dir] [output.md]
#
# Defaults: _sources/ and Attribution.md relative to this script.

require "open3"

SOURCES_ROOT = File.expand_path(ARGV[0] || File.join(ROOT, "_sources"))
OUTPUT       = File.expand_path(ARGV[1] || File.join(ROOT, "Attribution.md"))

LICENSE_GLOB = "{LICENSE,LICENCE,MIT-LICENSE,MIT-LICENCE,COPYING,COPYRIGHT}*"

# ---------------------------------------------------------------------------
# Repo discovery
# ---------------------------------------------------------------------------

def discover_repos(sources_root)
  Dir.glob(File.join(sources_root, "*"))
     .select { |p| File.directory?(p) }
     .reject { |p| File.basename(p).start_with?(".") }
     .sort
     .map { |p| [File.basename(p), p] }
end

# Origin URL from the git remote ("git@github.com:owner/repo.git" →
# "https://github.com/owner/repo"). nil if there is no remote.
def repo_url(repo_root)
  out, = Open3.capture2("git", "-C", repo_root, "remote", "get-url", "origin")
  url = out.strip
  return nil if url.empty?

  url.sub(%r{\Agit@([^:]+):}, "https://\\1/")
     .sub(/\.git\z/, "")
rescue StandardError
  nil
end

def license_file(repo_root)
  Dir.glob(File.join(repo_root, LICENSE_GLOB))
     .select { |f| File.file?(f) }
     .sort
     .first
end

# ---------------------------------------------------------------------------
# License detection (heuristic, order matters: specific before generic)
# ---------------------------------------------------------------------------

def detect_license(text)
  t = text[0, 8192]
  if t =~ /o['’]saasy/i
    "O'Saasy License"
  elsif t =~ /creative commons attribution[- ]sharealike/i
    version = t[/creative commons attribution[- ]sharealike\s+(\d+(?:\.\d+)?)/i, 1]
    version ? "CC BY-SA #{version}" : "CC BY-SA"
  elsif t =~ /apache license/i
    "Apache-2.0"
  elsif t =~ /gnu lesser general public license/i
    t =~ /version 3/i ? "LGPL-3.0" : (t =~ /version 2/i ? "LGPL-2.1" : "LGPL")
  elsif t =~ /gnu general public license/i
    t =~ /version 3/i ? "GPL-3.0" : (t =~ /version 2/i ? "GPL-2.0" : "GPL")
  elsif t =~ /mozilla public license/i
    "MPL-2.0"
  elsif t =~ /redistribution and use in source and binary forms/i
    t =~ /neither the name/i ? "BSD-3-Clause" : "BSD-2-Clause"
  elsif t =~ /permission is hereby granted, free of charge/i
    "MIT"
  elsif t =~ /do what the fuck you want/i
    "WTFPL"
  elsif t =~ /the unlicense/i
    "Unlicense"
  else
    nil
  end
end

# First copyright line in the license text, cleaned of prefixes/trailing dot.
def copyright_line(text)
  line = text.lines.first(25).find { |l| l =~ /copyright/i }
  return nil unless line

  line = line.sub(/\A#+\s*/, "").sub(/\A\*\*\s*/, "").strip
  line = line.sub(/\Acopyright\s*(?:\(c\)|©)?\s*/i, "").sub(/\.\z/, "").strip
  line.empty? ? nil : line
end

def build_table(rows)
  lines = []
  lines << "| Project | License | Copyright | Includes |"
  lines << "|---|---|---|---|"
  rows.each do |r|
    project = r[:url] ? "[#{r[:name]}](#{r[:url]})" : "`#{r[:name]}`"
    lines << "| #{project} | #{r[:license]} | #{r[:copyright]} | #{r[:includes]} |"
  end
  lines.join("\n")
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main
  abort "Sources dir not found: #{SOURCES_ROOT}" unless File.directory?(SOURCES_ROOT)

  rows = []
  discover_repos(SOURCES_ROOT).each do |name, repo_root|
    lic_path = license_file(repo_root)
    if lic_path.nil?
      warn "  [warn] #{name}: no license file found — skipped"
      next
    end

    text      = File.read(lic_path, encoding: "UTF-8")
    license   = detect_license(text) || "Other (see license file)"
    copyright = copyright_line(text) || "—"
    includes  = File.directory?(File.join(repo_root, "guides", "source")) ? "code + guides" : "code"

    rows << { name: name, url: repo_url(repo_root), license: license, copyright: copyright, includes: includes }
    puts "  #{name}: #{license} — #{copyright}"
  end

  abort "No attributable repositories found under #{SOURCES_ROOT}" if rows.empty?

  content = [
    "# Attribution — Sources Used to Generate This Dataset",
    "",
    "Generated by `build_attribution.rb` from the license files inside each",
    "repository under `_sources/` — do not edit by hand. Regenerate with:",
    "",
    "    ruby build/build_attribution.rb",
    "",
    build_table(rows),
    "",
  ].join("\n")

  File.write(OUTPUT, content, encoding: "UTF-8")
  puts "Wrote #{OUTPUT} with attribution for #{rows.size} repositories."
end

main if __FILE__ == $PROGRAM_NAME
