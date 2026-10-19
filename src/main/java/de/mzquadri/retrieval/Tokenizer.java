package de.mzquadri.retrieval;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Lowercase, join thousands separators, split on runs of non-alphanumerics.
 *
 * <p>A port of {@code src/retrieval.py:tokenize}. The two must agree exactly,
 * because every BM25 score downstream is a function of the token stream, so a
 * single divergence here moves rankings rather than producing a visible error.
 *
 * <p>The thousands rule exists so that "$30,000" and "30000" are the same term.
 * The NFIP forms write the Increased Cost of Compliance limit as "$30,000", and
 * a question asking about "30000" would otherwise share no term with the clause
 * that answers it.
 */
public final class Tokenizer {

    /**
     * A comma between a digit and exactly three digits followed by a word
     * boundary, which is the separator inside a number rather than punctuation
     * between two of them. Identical to the Python pattern, and Java supports
     * both the lookbehind and the lookahead as written.
     */
    private static final Pattern THOUSANDS = Pattern.compile("(?<=\\d),(?=\\d{3}\\b)");

    /** Runs of ASCII letters and digits. Applied after lowercasing. */
    private static final Pattern TOKEN = Pattern.compile("[a-z0-9]+");

    private Tokenizer() {
    }

    /**
     * @param text any text, document or query
     * @return its tokens, in order, duplicates kept
     */
    public static List<String> tokenize(String text) {
        // Locale.ROOT, not the default locale. A JVM running under a Turkish
        // locale lowercases 'I' to a dotless 'ı', which is not [a-z] and would
        // drop every token containing it. The Python side has no such notion,
        // so the locale has to be pinned rather than inherited.
        String lowered = text.toLowerCase(Locale.ROOT);
        String joined = THOUSANDS.matcher(lowered).replaceAll("");

        List<String> tokens = new ArrayList<>();
        Matcher matcher = TOKEN.matcher(joined);
        while (matcher.find()) {
            tokens.add(matcher.group());
        }
        return tokens;
    }
}
