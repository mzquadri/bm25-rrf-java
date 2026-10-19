package de.mzquadri.retrieval;

/**
 * One retrieved chunk and the score it was retrieved with.
 *
 * <p>The score's meaning depends on what produced it: a BM25 relevance score, a
 * cosine similarity, or a fused reciprocal-rank total. They are not comparable
 * across retrievers, which is the reason {@link ReciprocalRankFusion} combines
 * positions instead of scores.
 */
public record Hit(String chunkId, double score) {
}
