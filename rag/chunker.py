"""
Hierarchical markdown chunker that preserves document structure and metadata.

Key precision features:
- Preserves section hierarchy (headers) as metadata on each chunk
- Maintains table integrity (never splits mid-table)
- Overlap between chunks for context continuity
- Source tracking for citation grounding
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from config import ChunkConfig


@dataclass
class Chunk:
    """A text chunk with rich metadata for precise retrieval."""

    text: str
    chunk_id: str
    source_file: str
    section_hierarchy: list[str]  # e.g. ["NOTE 1", "Organization", "(a) Nature"]
    chunk_index: int  # Position within document
    total_chunks: int  # Total chunks from this document
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_file": self.source_file,
            "section_hierarchy": " > ".join(self.section_hierarchy),
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            **self.metadata,
        }


class MarkdownChunker:
    """Chunks markdown documents while preserving structural context."""

    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    TABLE_ROW_PATTERN = re.compile(r"^\|.*\|$", re.MULTILINE)

    def __init__(self, config: ChunkConfig | None = None):
        self.config = config or ChunkConfig()

    def chunk_file(self, file_path: Path) -> list[Chunk]:
        """Chunk a single markdown file."""
        text = file_path.read_text(encoding="utf-8")
        source_name = file_path.stem
        sections = self._split_into_sections(text)
        chunks: list[Chunk] = []

        for section_hierarchy, section_text in sections:
            section_chunks = self._split_section(section_text)
            for chunk_text in section_chunks:
                if len(chunk_text.strip()) < self.config.min_chunk_size:
                    continue
                chunks.append(
                    Chunk(
                        text=chunk_text.strip(),
                        chunk_id="",  # Will be set after total_chunks is known
                        source_file=source_name,
                        section_hierarchy=list(section_hierarchy),
                        chunk_index=len(chunks),
                        total_chunks=0,
                    )
                )

        # Set total_chunks and generate deterministic IDs
        for i, chunk in enumerate(chunks):
            chunk.total_chunks = len(chunks)
            chunk.chunk_index = i
            chunk.chunk_id = self._generate_id(source_name, i, chunk.text)

        return chunks

    def chunk_directory(self, dir_path: Path) -> list[Chunk]:
        """Chunk all markdown files in a directory."""
        all_chunks: list[Chunk] = []
        md_files = sorted(dir_path.glob("*.md"))

        for md_file in md_files:
            file_chunks = self.chunk_file(md_file)
            all_chunks.extend(file_chunks)
            print(f"  Chunked {md_file.name}: {len(file_chunks)} chunks")

        return all_chunks

    def _split_into_sections(self, text: str) -> list[tuple[list[str], str]]:
        """Split markdown into sections based on headers, preserving hierarchy."""
        sections: list[tuple[list[str], str]] = []
        current_hierarchy: list[str] = []
        current_text_parts: list[str] = []
        current_level = 0

        lines = text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]
            header_match = self.HEADER_PATTERN.match(line)

            if header_match:
                # Save accumulated text as a section
                if current_text_parts:
                    section_text = "\n".join(current_text_parts).strip()
                    if section_text:
                        sections.append((list(current_hierarchy), section_text))
                    current_text_parts = []

                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # Update hierarchy
                if level > current_level:
                    current_hierarchy.append(title)
                elif level == current_level:
                    if current_hierarchy:
                        current_hierarchy[-1] = title
                    else:
                        current_hierarchy.append(title)
                else:
                    # Go back up the hierarchy
                    trim_to = max(0, level - 1)
                    current_hierarchy = current_hierarchy[:trim_to]
                    current_hierarchy.append(title)

                current_level = level
                # Include header in the section text for context
                current_text_parts.append(line)
            else:
                current_text_parts.append(line)

            i += 1

        # Don't forget the last section
        if current_text_parts:
            section_text = "\n".join(current_text_parts).strip()
            if section_text:
                sections.append((list(current_hierarchy), section_text))

        # If no headers found, return entire text as one section
        if not sections:
            sections.append(([], text.strip()))

        return sections

    def _split_section(self, text: str) -> list[str]:
        """Split a section into chunks respecting tables and size limits."""
        if len(text) <= self.config.chunk_size:
            return [text]

        # Identify table blocks to avoid splitting them
        blocks = self._extract_blocks(text)
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_size = 0

        for block in blocks:
            block_size = len(block)

            if current_size + block_size <= self.config.chunk_size:
                current_chunk.append(block)
                current_size += block_size
            else:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))

                # If a single block exceeds chunk_size, force-split it
                if block_size > self.config.chunk_size:
                    sub_chunks = self._force_split(block)
                    chunks.extend(sub_chunks)
                    current_chunk = []
                    current_size = 0
                else:
                    current_chunk = [block]
                    current_size = block_size

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        # Apply overlap
        if len(chunks) > 1:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _extract_blocks(self, text: str) -> list[str]:
        """Extract logical blocks (paragraphs, tables) from text."""
        blocks: list[str] = []
        current_block_lines: list[str] = []
        in_table = False

        for line in text.split("\n"):
            is_table_line = bool(self.TABLE_ROW_PATTERN.match(line.strip()))

            if is_table_line:
                if not in_table and current_block_lines:
                    blocks.append("\n".join(current_block_lines))
                    current_block_lines = []
                in_table = True
                current_block_lines.append(line)
            else:
                if in_table:
                    blocks.append("\n".join(current_block_lines))
                    current_block_lines = []
                    in_table = False

                if line.strip() == "":
                    if current_block_lines:
                        blocks.append("\n".join(current_block_lines))
                        current_block_lines = []
                else:
                    current_block_lines.append(line)

        if current_block_lines:
            blocks.append("\n".join(current_block_lines))

        return blocks

    def _force_split(self, text: str) -> list[str]:
        """Force-split oversized text using sentence boundaries."""
        chunks: list[str] = []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        current: list[str] = []
        current_size = 0

        for sentence in sentences:
            if current_size + len(sentence) > self.config.chunk_size and current:
                chunks.append(" ".join(current))
                # Keep overlap
                overlap_text = " ".join(current)
                overlap_start = max(0, len(overlap_text) - self.config.chunk_overlap)
                overlap = overlap_text[overlap_start:]
                current = [overlap, sentence] if overlap else [sentence]
                current_size = sum(len(s) for s in current)
            else:
                current.append(sentence)
                current_size += len(sentence)

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap between adjacent chunks for context continuity."""
        if self.config.chunk_overlap <= 0:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap = prev[-self.config.chunk_overlap :]
            # Find a clean break point (start of word/sentence)
            space_idx = overlap.find(" ")
            if space_idx > 0:
                overlap = overlap[space_idx + 1 :]
            result.append(f"{overlap}\n{chunks[i]}")

        return result

    @staticmethod
    def _generate_id(source: str, index: int, text: str) -> str:
        """Generate a deterministic chunk ID."""
        content = f"{source}::{index}::{text[:100]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
