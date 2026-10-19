package de.mzquadri.retrieval;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class ChunkerTest {

    @Test
    @DisplayName("offsets address the original text, so a chunk can be cited back")
    void offsetsAddressTheSource() {
        String text = "   Coverage D pays up to $30,000 for compliance work.   ";
        Document document = new Document("doc", text);

        List<Chunk> chunks = Chunker.chunkDocument(document, 30, 5);

        assertTrue(chunks.size() > 1, "expected several windows");
        for (Chunk chunk : chunks) {
            assertEquals(
                    chunk.text(),
                    text.substring(chunk.start(), chunk.end()),
                    "text and offsets disagree for " + chunk.chunkId());
        }
    }

    @Test
    @DisplayName("the leading whitespace of a window is trimmed and the offset moved with it")
    void trimsAndShiftsOffsets() {
        Document document = new Document("doc", "     hello");

        List<Chunk> chunks = Chunker.chunkDocument(document, 10, 2);

        assertEquals(1, chunks.size());
        assertEquals("hello", chunks.get(0).text());
        assertEquals(5, chunks.get(0).start());
        assertEquals(10, chunks.get(0).end());
    }

    @Test
    @DisplayName("an all-whitespace window yields no chunk and does not consume an index")
    void skipsBlankWindows() {
        // Indices must stay contiguous: a gap would misreport chunk_index, and
        // in the Python original the same rule keeps the two sides aligned.
        Document document = new Document("doc", "ab" + "          " + "cd");

        List<Chunk> chunks = Chunker.chunkDocument(document, 4, 1);

        for (int i = 0; i < chunks.size(); i++) {
            assertEquals(i, chunks.get(i).chunkIndex());
        }
        for (Chunk chunk : chunks) {
            assertFalseBlank(chunk.text());
        }
    }

    @Test
    @DisplayName("the final partial window is kept")
    void keepsTheTail() {
        Document document = new Document("doc", "abcdefg");

        List<Chunk> chunks = Chunker.chunkDocument(document, 3, 1);

        String last = chunks.get(chunks.size() - 1).text();
        assertTrue("abcdefg".endsWith(last), "the tail was dropped; last chunk was " + last);
        assertEquals(7, chunks.get(chunks.size() - 1).end());
    }

    @Test
    @DisplayName("identical text in two documents gets different ids")
    void idsDoNotCollideAcrossDocuments() {
        // The three NFIP forms share passages verbatim. Hashing the text alone
        // would merge them into one chunk and lose a whole document's evidence.
        String shared = "Proof of loss within 60 days.";
        String a = Chunker.makeChunkId("dwelling", 0, shared);
        String b = Chunker.makeChunkId("rcbap", 0, shared);

        assertNotEquals(a, b);
        assertTrue(a.startsWith("dwelling#"));
        assertTrue(b.startsWith("rcbap#"));
    }

    @Test
    @DisplayName("identical text at a different offset gets a different id")
    void idsDependOnOffset() {
        String shared = "Proof of loss within 60 days.";
        assertNotEquals(
                Chunker.makeChunkId("doc", 0, shared),
                Chunker.makeChunkId("doc", 800, shared));
    }

    @Test
    @DisplayName("a configuration that could not terminate is rejected")
    void rejectsNonTerminatingConfigurations() {
        // The cursor advances by size - overlap, so overlap >= size would loop
        // forever rather than fail.
        assertThrows(IllegalArgumentException.class, () -> Chunker.validate(100, 100));
        assertThrows(IllegalArgumentException.class, () -> Chunker.validate(100, 200));
        assertThrows(IllegalArgumentException.class, () -> Chunker.validate(0, 0));
        assertThrows(IllegalArgumentException.class, () -> Chunker.validate(100, -1));
    }

    @Test
    @DisplayName("Python's whitespace set is used, not Java's")
    void stripsPythonWhitespace() {
        // Spelled as escapes rather than pasted, because these characters are
        // invisible in a diff and a reviewer cannot tell a no-break space from a
        // space by looking.
        final char noBreakSpace = '\u00A0';
        final char nextLine = '\u0085';
        final char lineSeparator = '\u2028';
        final char zeroWidthSpace = '\u200B';

        // Whitespace to Python, not to Character.isWhitespace, so Java's
        // String.strip() would leave these in place and shift every later offset.
        assertTrue(PythonText.isPythonSpace(noBreakSpace), "no-break space");
        assertTrue(PythonText.isPythonSpace(nextLine), "next line");
        assertTrue(PythonText.isPythonSpace(lineSeparator), "line separator");
        assertEquals("x", PythonText.strip(noBreakSpace + "x" + noBreakSpace));

        // Not whitespace to Python, so it must survive stripping.
        assertFalse(PythonText.isPythonSpace(zeroWidthSpace), "zero-width space");
        assertEquals(
                zeroWidthSpace + "x",
                PythonText.strip(" " + zeroWidthSpace + "x "));
    }

    private static void assertFalseBlank(String text) {
        assertEquals(text, PythonText.strip(text), "chunk text should already be stripped");
        assertTrue(!text.isEmpty(), "a blank chunk should not have been emitted");
    }
}
