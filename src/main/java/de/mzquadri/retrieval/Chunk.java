package de.mzquadri.retrieval;

/**
 * A window of one document, with the offsets it came from.
 *
 * <p>{@code document.text().substring(start, end)} is exactly {@code text}:
 * the chunker trims whitespace off each window and moves the offsets to match,
 * rather than pairing trimmed text with untrimmed offsets. A citation can
 * therefore be resolved back to a character range in the source document.
 */
public record Chunk(
        String chunkId,
        String documentId,
        int chunkIndex,
        int start,
        int end,
        String text) {
}
