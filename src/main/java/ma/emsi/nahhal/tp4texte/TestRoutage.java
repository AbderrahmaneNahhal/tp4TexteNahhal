package ma.emsi.nahhal.tp4texte;

import dev.langchain4j.data.document.Document;
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
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class TestRoutage {

    public static void main(String[] args) {
        String key = System.getenv("GEMINI_KEY");

        if (key == null || key.isBlank()) {
            System.err.println("La variable GEMINI_KEY n'est pas définie.");
            return;
        }

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(key)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        EmbeddingModel embeddingModel = new BgeSmallEnV15EmbeddingModel();

        // Chaque document possède sa propre base vectorielle.
        ContentRetriever retrieverIA =
                creerRetriever("rag.pdf", embeddingModel);

        ContentRetriever retrieverTourisme =
                creerRetriever("tourisme.txt", embeddingModel);

        // Les descriptions permettent au LLM de choisir la source.
        Map<ContentRetriever, String> sources = new LinkedHashMap<>();

        sources.put(retrieverIA,
                "Support de cours sur l'intelligence artificielle : "
                        + "RAG, fine-tuning, modèles de langage et recherche documentaire.");

        sources.put(retrieverTourisme,
                "Guide touristique contenant des lieux à visiter "
                        + "à Paris, Marrakech et Rome.");

        LanguageModelQueryRouter router =
                new LanguageModelQueryRouter(model, sources);

        DefaultRetrievalAugmentor augmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(router)
                        .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(augmentor)
                .build();

        System.out.println("Indexation des deux documents terminée.");
        conversationAvec(assistant);
    }

    private static ContentRetriever creerRetriever(
            String fichier, EmbeddingModel embeddingModel) {

        Document document = ClassPathDocumentLoader.loadDocument(
                fichier, new ApacheTikaDocumentParser());

        List<TextSegment> segments = DocumentSplitters
                .recursive(1200, 150)
                .split(document);

        List<Embedding> embeddings =
                embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> store =
                new InMemoryEmbeddingStore<>();

        store.addAll(embeddings, segments);

        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
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