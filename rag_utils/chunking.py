"""Text chunking utilities for optimal passage retrieval."""

from typing import List, Optional, Callable
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    start_idx: int
    end_idx: int
    source_id: Optional[str] = None
    metadata: Optional[dict] = None


class BaseChunker:
    """Base class for all chunking strategies."""

    def chunk(self, text: str, source_id: Optional[str] = None) -> List[Chunk]:
        """Chunk text into passages."""
        raise NotImplementedError


class FixedSizeChunker(BaseChunker):
    """
    Simple fixed-size chunking with optional overlap.
    Fast but may split in middle of sentences.
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize fixed-size chunker.

        Args:
            chunk_size: Size of each chunk in characters
            overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source_id: Optional[str] = None) -> List[Chunk]:
        """Chunk text with fixed size and overlap."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():  # Skip empty chunks
                chunks.append(Chunk(
                    text=chunk_text,
                    start_idx=start,
                    end_idx=end,
                    source_id=source_id
                ))

            start = end - self.overlap

        return chunks


class SentenceChunker(BaseChunker):
    """
    Chunks text by sentences, respecting sentence boundaries.
    Better than fixed-size for semantic coherence.
    """

    def __init__(self, max_sentences: int = 5, overlap_sentences: int = 1):
        """
        Initialize sentence-based chunker.

        Args:
            max_sentences: Maximum sentences per chunk
            overlap_sentences: Number of overlapping sentences between chunks
        """
        self.max_sentences = max_sentences
        self.overlap_sentences = overlap_sentences

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Handle common sentence endings
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str, source_id: Optional[str] = None) -> List[Chunk]:
        """Chunk text by sentences."""
        sentences = self._split_sentences(text)
        chunks = []

        i = 0
        while i < len(sentences):
            # Take max_sentences
            chunk_sentences = sentences[i:i + self.max_sentences]
            chunk_text = ' '.join(chunk_sentences)

            # Find positions in original text
            start_idx = text.find(chunk_sentences[0])
            end_idx = start_idx + len(chunk_text)

            chunks.append(Chunk(
                text=chunk_text,
                start_idx=start_idx,
                end_idx=end_idx,
                source_id=source_id
            ))

            # Move forward with overlap
            i += self.max_sentences - self.overlap_sentences

        return chunks


class RecursiveChunker(BaseChunker):
    """
    Recursive chunking that tries to split on natural boundaries.
    Tries paragraph > sentence > word boundaries in order.

    Inspired by LangChain's RecursiveCharacterTextSplitter.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize recursive chunker.

        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks
            separators: List of separators in order of preference
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Default separators in order of preference
        if separators is None:
            self.separators = [
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence ends
                "! ",
                "? ",
                "; ",
                ", ",    # Clause breaks
                " ",     # Word boundaries
                ""       # Character-level fallback
            ]
        else:
            self.separators = separators

    def _split_text(self, text: str, separator: str) -> List[str]:
        """Split text by separator."""
        if separator:
            return text.split(separator)
        else:
            return list(text)  # Character-level

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits into chunks of appropriate size."""
        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_length = len(split)

            if current_length + split_length + len(separator) > self.chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunk_text = separator.join(current_chunk)
                    if chunk_text:
                        chunks.append(chunk_text)

                    # Start new chunk with overlap
                    overlap_idx = 0
                    overlap_length = 0
                    for i in range(len(current_chunk) - 1, -1, -1):
                        overlap_length += len(current_chunk[i]) + len(separator)
                        if overlap_length >= self.overlap:
                            overlap_idx = i
                            break

                    current_chunk = current_chunk[overlap_idx:]
                    current_length = sum(len(s) for s in current_chunk) + len(separator) * (len(current_chunk) - 1)

            current_chunk.append(split)
            current_length += split_length + len(separator)

        # Add final chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    def chunk(self, text: str, source_id: Optional[str] = None) -> List[Chunk]:
        """Recursively chunk text using preferred separators."""
        # Try each separator in order
        for separator in self.separators:
            if separator in text or separator == "":
                splits = self._split_text(text, separator)

                # If splits are still too large, recurse with next separator
                good_splits = []
                for split in splits:
                    if len(split) <= self.chunk_size:
                        good_splits.append(split)
                    else:
                        # Need to split further
                        # Try next separator
                        continue

                # If all splits are good size, merge them
                if len(good_splits) == len(splits):
                    merged = self._merge_splits(splits, separator)

                    # Convert to Chunk objects
                    chunks = []
                    current_pos = 0
                    for chunk_text in merged:
                        start_idx = text.find(chunk_text, current_pos)
                        end_idx = start_idx + len(chunk_text)

                        chunks.append(Chunk(
                            text=chunk_text,
                            start_idx=start_idx,
                            end_idx=end_idx,
                            source_id=source_id
                        ))
                        current_pos = end_idx

                    return chunks

        # Fallback: just do fixed-size chunking
        return FixedSizeChunker(self.chunk_size, self.overlap).chunk(text, source_id)


class SemanticChunker(BaseChunker):
    """
    Semantic chunking using sentence embeddings.
    Groups sentences with similar embeddings together.

    More computationally expensive but produces semantically coherent chunks.
    """

    def __init__(
        self,
        embedding_model,
        max_chunk_size: int = 1000,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize semantic chunker.

        Args:
            embedding_model: Sentence transformer model
            max_chunk_size: Maximum chunk size in characters
            similarity_threshold: Cosine similarity threshold for grouping
        """
        self.embedding_model = embedding_model
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, a, b):
        """Compute cosine similarity between two vectors."""
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def chunk(self, text: str, source_id: Optional[str] = None) -> List[Chunk]:
        """Chunk text semantically by grouping similar sentences."""
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        # Encode all sentences
        embeddings = self.embedding_model.encode(sentences)

        # Group sentences by semantic similarity
        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_chunk_length = len(sentences[0])

        for i in range(1, len(sentences)):
            # Compute similarity with current chunk's last sentence
            similarity = self._cosine_similarity(embeddings[i-1], embeddings[i])

            # Check if we should add to current chunk or start new one
            would_exceed = current_chunk_length + len(sentences[i]) + 1 > self.max_chunk_size
            is_dissimilar = similarity < self.similarity_threshold

            if would_exceed or is_dissimilar:
                # Finalize current chunk
                chunk_text = ' '.join(current_chunk_sentences)
                start_idx = text.find(current_chunk_sentences[0])
                end_idx = start_idx + len(chunk_text)

                chunks.append(Chunk(
                    text=chunk_text,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    source_id=source_id
                ))

                # Start new chunk
                current_chunk_sentences = [sentences[i]]
                current_chunk_length = len(sentences[i])
            else:
                # Add to current chunk
                current_chunk_sentences.append(sentences[i])
                current_chunk_length += len(sentences[i]) + 1

        # Add final chunk
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            start_idx = text.find(current_chunk_sentences[0])
            end_idx = start_idx + len(chunk_text)

            chunks.append(Chunk(
                text=chunk_text,
                start_idx=start_idx,
                end_idx=end_idx,
                source_id=source_id
            ))

        return chunks


class DocumentChunker:
    """
    High-level chunker that processes multiple documents.
    Maintains document boundaries and metadata.
    """

    def __init__(self, chunker: BaseChunker):
        """
        Initialize document chunker.

        Args:
            chunker: Base chunking strategy to use
        """
        self.chunker = chunker

    def chunk_documents(
        self,
        documents: List[str],
        document_ids: Optional[List[str]] = None
    ) -> List[Chunk]:
        """
        Chunk multiple documents.

        Args:
            documents: List of document texts
            document_ids: Optional list of document IDs

        Returns:
            List of chunks with source metadata
        """
        all_chunks = []

        for i, doc in enumerate(documents):
            doc_id = document_ids[i] if document_ids else f"doc_{i}"
            chunks = self.chunker.chunk(doc, source_id=doc_id)
            all_chunks.extend(chunks)

        return all_chunks


# Factory function for easy chunker creation
def create_chunker(
    strategy: str = "recursive",
    chunk_size: int = 1000,
    overlap: int = 200,
    **kwargs
) -> BaseChunker:
    """
    Factory function to create chunkers.

    Args:
        strategy: Chunking strategy ("fixed", "sentence", "recursive", "semantic")
        chunk_size: Target chunk size
        overlap: Overlap between chunks
        **kwargs: Additional strategy-specific parameters

    Returns:
        Chunker instance
    """
    if strategy == "fixed":
        return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)

    elif strategy == "sentence":
        max_sentences = kwargs.get("max_sentences", 5)
        overlap_sentences = kwargs.get("overlap_sentences", 1)
        return SentenceChunker(max_sentences=max_sentences, overlap_sentences=overlap_sentences)

    elif strategy == "recursive":
        separators = kwargs.get("separators", None)
        return RecursiveChunker(chunk_size=chunk_size, overlap=overlap, separators=separators)

    elif strategy == "semantic":
        embedding_model = kwargs.get("embedding_model")
        if not embedding_model:
            raise ValueError("Semantic chunker requires 'embedding_model' parameter")
        similarity_threshold = kwargs.get("similarity_threshold", 0.7)
        return SemanticChunker(
            embedding_model=embedding_model,
            max_chunk_size=chunk_size,
            similarity_threshold=similarity_threshold
        )

    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


__all__ = [
    "Chunk",
    "BaseChunker",
    "FixedSizeChunker",
    "SentenceChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "DocumentChunker",
    "create_chunker"
]
