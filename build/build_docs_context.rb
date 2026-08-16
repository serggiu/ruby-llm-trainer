#!/usr/bin/env ruby
# frozen_string_literal: true

# Data root — where _sources/, _dataset/, etc. live. Override with
# LLM_TRAINER_ROOT (used by tests to run the pipeline in a sandbox).
ROOT = ENV["LLM_TRAINER_ROOT"] || File.expand_path("..", File.dirname(__FILE__)) unless defined?(ROOT)

# Build docs-context Markdown files from every repository under _sources/
# that ships guides.
#
# Source-agnostic: it scans _sources/*, runs `git pull` inside each repo (a
# failed pull is only a warning), and converts any repo that has a
# a `guides/source/documents.yaml` layout. Repos
# without guides are skipped. Output is namespaced per repository:
#
#   docs_context/<repo>/<category>/<guide>.md
#   docs_context/INDEX.md
#
# Stale output folders (for repos that no longer exist under _sources/) are
# removed automatically.
#
# Usage:
#   ruby build/build_docs_context.rb [sources_dir]
#
# Defaults to _sources/ relative to this script.

require "fileutils"
require "yaml"
require "set"

POS_ARGS   = ARGV.reject { |a| a.start_with?("--") }
FLAGS      = ARGV.select { |a| a.start_with?("--") }

SOURCES_ROOT = File.expand_path(POS_ARGS[0] || File.join(ROOT, "_sources"))
OUTPUT_DIR   = File.join(ROOT, "docs_context")

# ---------------------------------------------------------------------------
# Repo discovery & git refresh
# ---------------------------------------------------------------------------

def discover_repos(sources_root)
  Dir.glob(File.join(sources_root, "*"))
     .select { |p| File.directory?(p) }
     .reject { |p| File.basename(p).start_with?(".") }
     .sort
     .map { |p| [File.basename(p), p] }
end

def git_pull(repo_root)
  name = File.basename(repo_root)
  puts "  Updating #{name} (git pull)..."
  Dir.chdir(repo_root) do
    ok = system({ "GIT_EDITOR" => "true" }, "git", "pull")
    warn "  [warn] `git pull` failed for #{name} — continuing with the existing sources (they may be stale)." unless ok
  end
end

# ---------------------------------------------------------------------------
# Release-notes filtering (source-agnostic)
# ---------------------------------------------------------------------------

RELEASE_NOTES_RE = /\A(\d+)_(\d+)_release_notes\z/

# Keeps only the newest release-notes guide per category (compared by version
# number) and returns the dropped guide URLs for reporting. Works for any repo
# whose guides follow the `<major>_<minor>_release_notes` naming — nothing is
# hardcoded to a specific version. Pass --all-release-notes to disable.
def filter_release_notes(categories)
  dropped = []
  categories.each do |cat|
    versioned = cat[:guides].select { |g| g[:url] =~ RELEASE_NOTES_RE }
    next if versioned.size <= 1

    latest = versioned.max_by do |g|
      m = g[:url].match(RELEASE_NOTES_RE)
      [m[1].to_i, m[2].to_i]
    end
    old = versioned - [latest]
    dropped.concat(old.map { |g| g[:url] })
    cat[:guides] -= old
  end
  dropped
end

# ---------------------------------------------------------------------------
# Parse documents.yaml into structured data
# ---------------------------------------------------------------------------

def parse_documents_yaml(path)
  raw = YAML.safe_load(File.read(path), permitted_classes: [Symbol])
  categories = []

  raw.each do |section|
    cat_name = section["name"]
    guides   = []

    section["documents"]&.each do |doc|
      name = doc["name"]
      next if name.nil? || name.strip.empty?

      guides << {
        title:            name,
        url:              doc["url"]&.sub(/\.html$/, ""),
        description:      doc["description"]&.strip,
        work_in_progress: doc["work_in_progress"] || false
      }
    end

    categories << { name: cat_name, guides: guides }
  end

  categories
end

# ---------------------------------------------------------------------------
# Find guides not listed in documents.yaml
# ---------------------------------------------------------------------------

def find_orphan_guides(guides_root, catalog_urls)
  Dir.glob(File.join(guides_root, "*.md")).map { |f| File.basename(f, ".md") }
     .reject { |slug| catalog_urls.include?(slug) }
     .sort
end

# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------

def slugify(str)
  str.downcase.gsub(/[^a-z0-9]+/, "-").gsub(/\A-|-\z/, "")
end

# ---------------------------------------------------------------------------
# Markdown processing
# ---------------------------------------------------------------------------

# YAML-safe string: wraps in quotes if it contains special characters
def yaml_safe(str)
  return '""' if str.nil? || str.empty?
  if str.match?(/[\n:"'#&*,?@\[\]{}!|>]|:\s|^\s|\s$/)
    str.inspect
  else
    str
  end
end

def build_front_matter(guide)
  lines = []
  lines << "---"
  lines << "title: #{yaml_safe(guide[:title])}"
  lines << "category: #{yaml_safe(guide[:category])}"
  lines << "description: #{yaml_safe(guide[:description])}" if guide[:description]
  lines << "work_in_progress: true" if guide[:work_in_progress]
  lines << "source_url: #{guide[:url]}.html" if guide[:url]
  lines << "---"
  lines << ""
  lines.join("\n")
end

def clean_guide_content(raw)
  # Strip the "DO NOT READ THIS FILE ON GITHUB" banner
  content = raw.sub(/\A\*\*DO NOT READ.*?\*+\n+/m, "")
  # Strip trailing whitespace but preserve intentional blank lines
  content.strip
end

# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def write_guide(output_dir, subdir, guide, content)
  dir = File.join(output_dir, subdir)
  FileUtils.mkdir_p(dir)

  filename = "#{guide[:url]}.md"
  path = File.join(dir, filename)

  front = build_front_matter(guide)
  File.write(path, front + content, encoding: "UTF-8")
  path
end

def build_index(repo_data, total_count, total_wip)
  lines = []
  lines << "# Docs Context — Index"
  lines << ""
  lines << "Guides converted from every repository under _sources/ that ships"
  lines << "a `guides/source/documents.yaml` layout, one folder per repository,"
  lines << "with YAML front matter for AI agent training."
  lines << ""
  lines << "**#{total_count} guides** (#{total_wip} WIP) from #{repo_data.size} repositor#{repo_data.size == 1 ? 'y' : 'ies'}."
  lines << ""
  lines << "---"
  lines << ""

  repo_data.each do |entry|
    repo = entry[:repo]
    lines << "## #{repo}"
    lines << ""

    entry[:categories].each do |cat|
      next if cat[:guides].empty?
      lines << "### #{cat[:name]}"
      lines << ""
      cat[:guides].each do |g|
        wip = g[:work_in_progress] ? " ⚠️ WIP" : ""
        lines << "- **[#{g[:title]}](#{repo}/#{slugify(cat[:name])}/#{g[:url]}.md)**#{wip}"
        lines << "  #{g[:description]}" if g[:description]
      end
      lines << ""
    end

    unless entry[:orphan_slugs].empty?
      lines << "### Additional Guides"
      lines << ""
      entry[:orphan_slugs].each do |slug|
        title = slug.tr("_", " ").split.map(&:capitalize).join(" ")
        lines << "- **[#{title}](#{repo}/additional/#{slug}.md)**"
      end
      lines << ""
    end
  end

  lines << "---"
  lines << ""
  lines << "## Usage"
  lines << ""
  lines << "To set up an AI agent with documentation context:"
  lines << ""
  lines << "1. Point the agent at `docs_context/INDEX.md` so it knows what's available."
  lines << "2. Have it read the `.md` files relevant to the task at hand."
  lines << "3. Combine with `code_context/` for source code and `AGENTS.md` for conventions."
  lines << ""
  lines << "Each guide file includes YAML front matter with `title`, `category`,"
  lines << "`description`, and `work_in_progress` fields for easy filtering."

  lines.join("\n")
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main
  abort "Sources dir not found: #{SOURCES_ROOT}" unless File.directory?(SOURCES_ROOT)

  repos = discover_repos(SOURCES_ROOT)
  abort "No repositories found under #{SOURCES_ROOT}" if repos.empty?

  FileUtils.mkdir_p(OUTPUT_DIR)

  repo_names = repos.map(&:first)

  # Remove output folders for repos that no longer exist under _sources/.
  Dir.glob(File.join(OUTPUT_DIR, "*")).select { |p| File.directory?(p) }.each do |dir|
    next if repo_names.include?(File.basename(dir))
    puts "  [clean] removing stale #{File.basename(dir)}/"
    FileUtils.rm_rf(dir)
  end

  repo_data = []
  total     = 0
  total_wip = 0
  written   = Set.new

  repos.each do |repo, repo_root|
    git_pull(repo_root)

    guides_root = File.join(repo_root, "guides", "source")
    unless File.directory?(guides_root)
      puts "  [skip] #{repo} — no guides/ directory"
      next
    end

    yaml_path = File.join(guides_root, "documents.yaml")
    unless File.exist?(yaml_path)
      puts "  [skip] #{repo} — guides/ found but no documents.yaml"
      next
    end

    puts "Processing guides for #{repo}..."
    categories = parse_documents_yaml(yaml_path)

    # Build the full URL set BEFORE filtering, so skipped release notes don't
    # resurface as orphans.
    catalog_urls = categories.flat_map { |c| c[:guides].map { |g| g[:url] } }.to_set

    unless FLAGS.include?("--all-release-notes")
      filter_release_notes(categories).each do |url|
        puts "  [skip] #{url}.md — older release notes (keeping only the latest)"
      end
    end

    # Find orphan .md files (against the FULL pre-filter catalog, so skipped
    # release notes don't resurface as orphans)
    orphan_slugs = find_orphan_guides(guides_root, catalog_urls)
    repo_count = 0
    repo_wip   = 0

    # Process cataloged guides grouped by category
    categories.each do |cat|
      cat_slug = slugify(cat[:name])
      cat[:guides].each do |guide|
        source_path = File.join(guides_root, "#{guide[:url]}.md")
        unless File.exist?(source_path)
          puts "  [warn] Missing source: #{guide[:url]}.md — skipping"
          next
        end

        raw     = File.read(source_path, encoding: "UTF-8")
        content = clean_guide_content(raw)
        guide_with_category = guide.merge(category: cat[:name])

        write_guide(OUTPUT_DIR, File.join(repo, cat_slug), guide_with_category, content)
        written << File.join(repo, cat_slug, "#{guide[:url]}.md")
        wip_tag = guide[:work_in_progress] ? " ⚠️" : ""
        puts "  #{repo}/#{cat_slug}/#{guide[:url]}.md#{wip_tag}"
        repo_count += 1
        repo_wip += 1 if guide[:work_in_progress]
      end
    end

    # Process orphan guides
    unless orphan_slugs.empty?
      puts "\n  Processing additional (uncataloged) guides for #{repo}..."
      orphan_slugs.each do |slug|
        source_path = File.join(guides_root, "#{slug}.md")
        raw     = File.read(source_path, encoding: "UTF-8")
        content = clean_guide_content(raw)
        title   = slug.tr("_", " ").split.map(&:capitalize).join(" ")

        guide = {
          title:            title,
          category:         "Additional Guides",
          url:              slug,
          description:      nil,
          work_in_progress: false
        }

        write_guide(OUTPUT_DIR, File.join(repo, "additional"), guide, content)
        written << File.join(repo, "additional", "#{slug}.md")
        puts "  #{repo}/additional/#{slug}.md"
        repo_count += 1
      end
    end

    repo_data << { repo: repo, categories: categories, orphan_slugs: orphan_slugs }
    total += repo_count
    total_wip += repo_wip
  end

  # Remove guide files that were not written this run (e.g. release notes
  # filtered out since the previous run).
  Dir.glob(File.join(OUTPUT_DIR, "**", "*.md")).each do |f|
    next if File.basename(f) == "INDEX.md"
    rel = f.sub("#{OUTPUT_DIR}/", "")
    next if written.include?(rel)
    puts "  [clean] removing stale #{rel}"
    File.delete(f)
  end

  # Write index
  puts "\n  Writing INDEX.md..."
  index = build_index(repo_data, total, total_wip)
  File.write(File.join(OUTPUT_DIR, "INDEX.md"), index, encoding: "UTF-8")

  puts "\nDone! #{total} guides (#{total_wip} WIP) from #{repo_data.size} repos written to docs_context/"
end

main if __FILE__ == $PROGRAM_NAME
