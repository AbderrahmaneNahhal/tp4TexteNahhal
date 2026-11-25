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
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class TestRechercheWeb {

    public static void main(String[] args) {
        String geminiKey = System.getenv("GEMINI_KEY");
        String tavilyKey = System.getenv("TAVILY_API_KEY");

        if (geminiKey == null || geminiKey.isBlank()) {
            System.err.println("La variable GEMINI_KEY n'est pas définie.");
            return;
        }

        if (tavilyKey == null || tavilyKey.isBlank()) {
            System.err.println("La variable TAVILY_API_KEY n'est pas définie.");
            return;
        }

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(geminiKey)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        EmbeddingModel embeddingModel = new BgeSmallEnV15EmbeddingModel();

        ContentRetriever retrieverPDF = creerRetrieverPDF(embeddingModel);

        WebSearchEngine moteurWeb = TavilyWebSearchEngine.builder()
                .apiKey(tavilyKey)
                .build();

        ContentRetriever retrieverWeb = WebSearchContentRetriever.builder()
                .webSearchEngine(moteurWeb)
                .maxResults(3)
                .build();

        /*
         * Ancienne version : recherche systématique dans les deux sources.
         *
         * DefaultQueryRouter router =
         *         new DefaultQueryRouter(retrieverPDF, retrieverWeb);
         *
         * Problème observé : des passages du cours sur le RAG étaient ajoutés
         * à une question sur le football, avec une confusion des sources
         * dans la réponse finale.
         */

        Map<ContentRetriever, String> sources = new LinkedHashMap<>();

        sources.put(retrieverPDF, """
                Support de cours sur le RAG, le fine-tuning et les modèles
                de langage. À utiliser pour les questions théoriques
                concernant ces thèmes ou le contenu du support.
                Ne contient aucune information sur le football ou l'actualité.
                """);

        sources.put(retrieverWeb, """
                Recherche sur le Web pour les actualités, les résultats
                sportifs, les événements récents et les informations
                absentes du support de cours.
                À utiliser pour les informations nécessitant une mise à jour,
                même si elles concernent l'intelligence artificielle.
                """);

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

        System.out.println("Indexation terminée : routage sélectif PDF / Web.");
        conversationAvec(assistant);
    }

    private static ContentRetriever creerRetrieverPDF(
            EmbeddingModel embeddingModel) {

        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf", new ApacheTikaDocumentParser());

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