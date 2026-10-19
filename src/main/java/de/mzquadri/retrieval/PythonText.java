package de.mzquadri.retrieval;

/**
 * Python's {@code str.strip()} semantics, which are not Java's.
 *
 * <p>The chunker trims each window and shifts its offsets by how much came off
 * the front, so the offsets and the chunk id both depend on exactly which
 * characters count as whitespace. Java's {@code String.strip()} uses
 * {@link Character#isWhitespace}, which disagrees with Python's
 * {@code str.isspace()} on characters that appear in real documents:
 *
 * <ul>
 *   <li>U+00A0 no-break space: Python strips it, {@code Character.isWhitespace}
 *       says false, so Java's {@code strip()} keeps it.</li>
 *   <li>U+0085 next line: the same disagreement.</li>
 *   <li>U+2007 figure space: wrong in the other direction -
 *       {@code Character.isSpaceChar} says true while Python says false, so
 *       reaching for that predicate instead would not fix it either.</li>
 * </ul>
 *
 * <p>The three NFIP forms happen to contain only U+0020 and U+000A, so for that
 * corpus {@code String.strip()} would have matched. This class exists so parity
 * is a property of the code rather than of one input: a corpus carrying a
 * no-break space would otherwise shift every offset after it and change every
 * chunk id downstream, which is a hard failure to read backwards from.
 */
public final class PythonText {

    private PythonText() {
    }

    /**
     * Whether Python's {@code str.isspace()} is true for this character.
     *
     * <p>Written out rather than delegated to a Java predicate because no Java
     * predicate matches Python's set.
     *
     * <p>Tab, line feed, form feed and carriage return use the short escapes
     * rather than unicode escapes. Java decodes backslash-u sequences before
     * the lexer runs, so a backslash-u000A here would become a real newline
     * inside a character literal and fail to compile.
     */
    public static boolean isPythonSpace(char c) {
        switch (c) {
            case '\t':         // U+0009 tab
            case '\n':         // U+000A line feed
            case '\u000B':     // vertical tab
            case '\f':         // U+000C form feed
            case '\r':         // U+000D carriage return
            case '\u001C':     // file separator
            case '\u001D':     // group separator
            case '\u001E':     // record separator
            case '\u001F':     // unit separator
            case '\u0020':     // space
            case '\u0085':     // next line
            case '\u00A0':     // no-break space
            case '\u1680':     // ogham space mark
            case '\u2028':     // line separator
            case '\u2029':     // paragraph separator
            case '\u202F':     // narrow no-break space
            case '\u205F':     // medium mathematical space
            case '\u3000':     // ideographic space
                return true;
            default:
                // U+2000..U+200A, the en quad through hair space. U+200B, the
                // zero-width space, sits just outside on purpose: Python does
                // not count it as whitespace.
                return c >= '\u2000' && c <= '\u200A';
        }
    }

    /** Index of the first character that is not Python-whitespace, or the length. */
    public static int firstNonSpace(String text) {
        int i = 0;
        while (i < text.length() && isPythonSpace(text.charAt(i))) {
            i++;
        }
        return i;
    }

    /** Index just past the last character that is not Python-whitespace. */
    public static int lastNonSpaceEnd(String text) {
        int i = text.length();
        while (i > 0 && isPythonSpace(text.charAt(i - 1))) {
            i--;
        }
        return i;
    }

    /** Equivalent of Python's {@code text.strip()}. */
    public static String strip(String text) {
        int start = firstNonSpace(text);
        if (start == text.length()) {
            return "";
        }
        return text.substring(start, lastNonSpaceEnd(text));
    }
}
