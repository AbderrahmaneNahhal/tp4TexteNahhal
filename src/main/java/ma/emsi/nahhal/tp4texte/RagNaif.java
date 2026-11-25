package ma.emsi.nahhal.tp4texte;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15.BgeSmallEnV15EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;
import java.util.Scanner;

public class RagNaif {

    public static void main(String[] args) {
        String apiKey = System.getenv("GEMINI_KEY");

        if (apiKey == null || apiKey.isBlank()) {
            System.err.println(
                    "La variable d'environnement GEMINI_KEY n'est pas définie.");
            return;
        }

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // PHASE 1 : préparer les documents et enregistrer les embeddings.

        System.out.println("Lecture du fichier rag.pdf...");

        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf", new ApacheTikaDocumentParser());

        // Sans tokenizer, ces deux tailles sont exprimées en caractères.
        DocumentSplitter splitter = DocumentSplitters.recursive(1200, 150);
        List<TextSegment> segments = splitter.split(document);

        System.out.println("Nombre de segments : " + segments.size());

        EmbeddingModel embeddingModel = new BgeSmallEnV15EmbeddingModel();

        List<Embedding> embeddings =
                embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        System.out.println("Indexation terminée.");

        // PHASE 2 : chercher les passages pertinents pour chaque question.

        EmbeddingStoreContentRetriever retriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .contentRetriever(retriever)
                .build();

        conversationAvec(assistant);
    }

    private static void conversationAvec(Assistant assistant) {
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question (ou tapez fin) :");

                if (!scanner.hasNextLine()) {
                    break;
                }

                String question = scanner.nextLine().trim();

                if ("fin".equalsIgnoreCase(question)) {
                    break;
                }

                if (question.isBlank()) {
                    continue;
                }

                String reponse = assistant.chat(question);
                System.out.println("Assistant : " + reponse);
            }
        }
    }
}
