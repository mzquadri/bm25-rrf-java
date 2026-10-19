package de.mzquadri.retrieval;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Locale;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** Tokenizer behaviour, stated independently of the Python fixture. */
class TokenizerTest {

    @Test
    @DisplayName("lowercases and splits on runs of non-alphanumerics")
    void splitsOnNonAlphanumerics() {
        assertEquals(
                List.of("flood", "insurance", "policy"),
                Tokenizer.tokenize("Flood  Insurance -- POLICY!"));
    }

    @Test
    @DisplayName("joins a thousands separator inside a number")
    void joinsThousandsSeparators() {
        assertEquals(List.of("30000"), Tokenizer.tokenize("$30,000"));
        assertEquals(List.of("1000000"), Tokenizer.tokenize("1,000,000"));
    }

    @Test
    @DisplayName("leaves a comma that is not a thousands separator alone")
    void leavesOtherCommasAlone() {
        // Two separate numbers in a list, not one number. The lookahead requires
        // exactly three digits then a word boundary, so "4,61" does not qualify.
        assertEquals(List.of("44", "61"), Tokenizer.tokenize("44,61"));
        assertEquals(List.of("a", "b"), Tokenizer.tokenize("a,b"));
    }

    @Test
    @DisplayName("keeps repeated terms, because BM25 scores them twice")
    void keepsDuplicates() {
        assertEquals(List.of("flood", "flood"), Tokenizer.tokenize("flood flood"));
    }

    @Test
    @DisplayName("drops characters outside a-z0-9, including the corpus punctuation")
    void dropsNonAsciiPunctuation() {
        // The NFIP forms contain em dashes and curly quotes. None of them are
        // [a-z0-9], so they act as separators rather than becoming tokens.
        List<String> tokens = Tokenizer.tokenize("“proof of loss” — 60 days");
        assertEquals(List.of("proof", "of", "loss", "60", "days"), tokens);
    }

    @Test
    @DisplayName("is not affected by the JVM's default locale")
    void isLocaleIndependent() {
        // A Turkish default locale lowercases 'I' to a dotless i, which is not
        // in [a-z] and would drop the token. Tokenizer pins Locale.ROOT; this
        // fails if that is ever removed.
        Locale original = Locale.getDefault();
        try {
            Locale.setDefault(new Locale("tr", "TR"));
            assertEquals(List.of("icc", "increased"), Tokenizer.tokenize("ICC Increased"));
        } finally {
            Locale.setDefault(original);
        }
    }

    @Test
    @DisplayName("returns nothing for text with no alphanumerics")
    void emptyForPunctuationOnly() {
        assertTrue(Tokenizer.tokenize("--- !!! ---").isEmpty());
        assertTrue(Tokenizer.tokenize("").isEmpty());
    }
}
