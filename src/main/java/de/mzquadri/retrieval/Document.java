package de.mzquadri.retrieval;

/** One source document, identified the way the corpus manifest identifies it. */
public record Document(String documentId, String text) {
}
