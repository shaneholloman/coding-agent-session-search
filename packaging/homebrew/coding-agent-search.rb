class CodingAgentSearch < Formula
  desc "Unified TUI search over local coding agent histories"
  homepage "https://github.com/coding-agent-search/coding-agent-search"
  version "0.1.0"
  url "https://github.com/coding-agent-search/coding-agent-search/releases/download/v0.1.0/coding-agent-search-v0.1.0-macos-arm64.tar.gz"
  # TODO: replace with real checksum when publishing releases
  sha256 "REPLACE_WITH_REAL_SHA256"
  license "MIT"

  def install
    bin.install "coding-agent-search"
    generate_completions_from_executable(bin/"coding-agent-search", "completions", shells: [:bash, :zsh, :fish])
    man1.install buildpath/"coding-agent-search.1" if File.exist?("coding-agent-search.1")
  end

  test do
    system "#{bin}/coding-agent-search", "--help"
  end
end
