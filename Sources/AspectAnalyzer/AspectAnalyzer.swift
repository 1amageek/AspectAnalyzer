import Foundation
import OllamaKit
import Logging

/// An actor that performs aspect analysis on queries.
///
/// `AspectAnalyzer` breaks down queries into their key aspects, evaluating their importance,
/// required knowledge areas, and expected information types. It also assesses query complexity
/// using configurable weights and thresholds.
///
/// Example usage:
/// ```swift
/// let analyzer = AspectAnalyzer(model: "llama3.2")
/// let analysis = try await analyzer.analyzeQuery(
///     "What is the impact of quantum computing on cryptography?"
/// )
///
/// // Access analysis results
/// print("Complexity score:", analysis.complexityScore)
/// print("Critical aspects:", analysis.criticalAspects)
/// ```
///
/// - Important: This actor is safe for concurrent use.
public struct AspectAnalyzer: Sendable {
    private let ollamaKit: OllamaKit
    private let logger: Logger?
    
    /// Represents a single aspect of a query with its characteristics and importance.
    ///
    /// Each aspect captures a specific facet of the query, including its importance level,
    /// the knowledge areas required to understand it, and the types of information expected.
    ///
    /// Example:
    /// ```swift
    /// let aspect = Aspect(
    ///     description: "Quantum computing fundamentals",
    ///     importance: 0.8,
    ///     requiredKnowledge: ["quantum_computing", "physics"],
    ///     expectedInfoTypes: ["technical", "theoretical"]
    /// )
    /// ```
    public struct Aspect: Codable, Equatable, Hashable, Sendable {
        /// A clear description of the aspect
        public let description: String
        
        /// The importance score of this aspect (0.0 to 1.0)
        ///
        /// Higher values indicate greater relevance to the query.
        public let importance: Float
        
        /// Set of knowledge areas required to understand this aspect
        public let requiredKnowledge: Set<String>
        
        /// Set of expected information types for this aspect
        ///
        /// Examples: "technical", "theoretical", "practical", etc.
        public let expectedInfoTypes: Set<String>
    }
    
    /// Results of query analysis containing aspects, complexity, and focus areas.
    ///
    /// This structure provides comprehensive analysis results, including extracted aspects,
    /// complexity assessment, and primary focus areas identified from the query.
    ///
    /// Example:
    /// ```swift
    /// let analysis = try await analyzer.analyzeQuery(query)
    /// print("Complexity:", analysis.complexityScore)
    /// for aspect in analysis.prioritizedAspects {
    ///     print("- \(aspect.description): \(aspect.importance)")
    /// }
    /// ```
    public struct Analysis: Sendable {
        /// The original query text that was analyzed
        public let query: String
        
        /// Array of extracted aspects with their evaluations
        public let aspects: [Aspect]
        
        /// Primary focus areas identified from the analysis
        public let primaryFocus: Set<String>
        
        /// Overall complexity score of the query (0.0 to 1.0)
        public let complexityScore: Float
        
        /// Returns aspects sorted by importance in descending order
        ///
        /// The most important aspects appear first in the resulting array.
        public var prioritizedAspects: [Aspect] {
            aspects.sorted { $0.importance > $1.importance }
        }
        
        /// Returns aspects with importance greater than 0.7
        ///
        /// These represent the most critical aspects of the query that require
        /// particular attention.
        public var criticalAspects: [Aspect] {
            aspects.filter { $0.importance > 0.7 }
        }
        
        public init(
            query: String,
            aspects: [Aspect],
            primaryFocus: Set<String>,
            complexityScore: Float
        ) {
            self.query = query
            self.aspects = aspects
            self.primaryFocus = primaryFocus
            self.complexityScore = complexityScore
        }
    }
    
    public let model: String
    
    /// Creates a new AspectAnalyzer instance.
    ///
    /// - Parameters:
    ///   - model: The identifier of the language model to use for analysis (default: "llama3.2:latest")
    ///   - logger: Optional logger for debug and error information
    public init(model: String = "llama3.2:latest", logger: Logger? = nil) {
        self.model = model
        self.logger = logger
        self.ollamaKit = OllamaKit()
    }
    
    /// Analyzes a query to identify and evaluate its aspects.
    ///
    /// This method performs comprehensive analysis of the query, breaking it down into
    /// aspects and evaluating their characteristics using the specified language model.
    ///
    /// - Parameters:
    ///   - query: The query string to analyze
    ///   - configuration: Configuration for complexity calculation (default: .default)
    /// - Returns: Analysis results containing aspects and evaluations
    /// - Throws: AnalysisError if the analysis fails
    public func analyzeQuery(_ query: String, configuration: ComplexityConfiguration = .default) async throws -> Analysis {
        logger?.debug("Starting query analysis", metadata: [
            "query": .string(query)
        ])
        
        let aspects = try await extractAspects(from: query)
        let complexity = calculateComplexity(query: query, aspects: aspects, configuration: configuration)
        let focusAreas = determinePrimaryFocus(aspects: aspects)
        
        logger?.info("Completed query analysis", metadata: [
            "aspectCount": .string("\(aspects.count)"),
            "complexity": .string(String(format: "%.2f", complexity)),
            "focusAreas": .string(focusAreas.joined(separator: ", "))
        ])
        
        return Analysis(
            query: query,
            aspects: aspects,
            primaryFocus: Set(focusAreas),
            complexityScore: complexity
        )
    }
    
    /// Extracts aspects from the query using LLM
    private func extractAspects(from query: String) async throws -> [Aspect] {
        let prompt = """
        Analyze the following query and identify its key aspects. For each aspect, determine its importance, required knowledge areas, and expected information types.
        
        Query: \(query)
        
        Provide the analysis in the following JSON format:
        {
            "aspects": [
                {
                    "description": "Clear description of the aspect",
                    "importance": 0.0-1.0,
                    "requiredKnowledge": ["area1", "area2"],
                    "expectedInfoTypes": ["type1", "type2"]
                }
            ]
        }
        
        Guidelines:
        - Break down complex queries into distinct aspects
        - Assign importance scores based on centrality to the query
        - Include specific knowledge areas needed
        - Specify types of information expected (e.g., technical, statistical, conceptual)
        """
        
        let data = OKChatRequestData(
            model: model,
            messages: [
                OKChatRequestData.Message(
                    role: .system,
                    content: """
                    You are a query analysis expert. Your task is to:
                    1. Break down queries into key aspects
                    2. Evaluate importance of each aspect
                    3. Identify required knowledge areas
                    4. Specify expected information types
                    Provide analysis in structured JSON format only.
                    """
                ),
                OKChatRequestData.Message(role: .user, content: prompt)
            ]
        ) { options in
            options.temperature = 0 // Deterministic output
            options.topP = 1
            options.topK = 1
        }
        
        // Collect response
        var response = ""
        for try await chunk in ollamaKit.chat(data: data) {
            response += chunk.message?.content ?? ""
        }
        
        // Clean up response to ensure valid JSON
        let jsonResponse = cleanJsonResponse(response)
        
        // Parse response
        guard let jsonData = jsonResponse.data(using: .utf8) else {
            throw AnalysisError.invalidResponse
        }
        
        struct Response: Codable, Sendable {
            let aspects: [Aspect]
        }
        
        do {
            let decoded = try JSONDecoder().decode(Response.self, from: jsonData)
            return decoded.aspects
        } catch {
            logger?.error("Failed to decode aspects", metadata: [
                "error": .string(error.localizedDescription),
                "response": .string(response)
            ])
            throw AnalysisError.decodingFailed(error)
        }
    }
    
    /// Determines primary focus areas from aspects
    private func determinePrimaryFocus(aspects: [Aspect]) -> [String] {
        // Collect all knowledge areas with their cumulative importance
        var areaImportance: [String: Float] = [:]
        
        for aspect in aspects {
            let importance = aspect.importance
            for area in aspect.requiredKnowledge {
                areaImportance[area, default: 0] += importance
            }
        }
        
        // Select areas with high cumulative importance
        let threshold: Float = 0.7
        return areaImportance
            .filter { $0.value >= threshold }
            .sorted { $0.value > $1.value }
            .map(\.key)
    }
    
    /// Cleans JSON response string to ensure validity
    private func cleanJsonResponse(_ response: String) -> String {
        // Extract JSON content between first { and last }
        if let start = response.firstIndex(of: "{"),
           let end = response.lastIndex(of: "}") {
            let jsonContent = response[start...end]
            return String(jsonContent)
        }
        return response
    }
}

// MARK: - Error Types

extension AspectAnalyzer {
    /// Errors that can occur during query analysis
    enum AnalysisError: Error {
        /// Response from LLM was invalid
        case invalidResponse
        /// Failed to decode response
        case decodingFailed(Error)
    }
}

/// Configuration for complexity calculation weights and thresholds
public struct ComplexityConfiguration: Sendable {
    /// Weights for different complexity factors (must sum to 1.0)
    public struct Weights: Sendable {
        /// Weight for aspect count factor (default: 0.3)
        public let aspectCount: Float
        /// Weight for average importance factor (default: 0.3)
        public let importance: Float
        /// Weight for knowledge breadth factor (default: 0.2)
        public let knowledge: Float
        /// Weight for information type diversity factor (default: 0.2)
        public let infoType: Float
        
        /// Validates that weights sum to 1.0
        public var isValid: Bool {
            abs((aspectCount + importance + knowledge + infoType) - 1.0) < Float.ulpOfOne
        }
        
        public init(
            aspectCount: Float = 0.3,
            importance: Float = 0.3,
            knowledge: Float = 0.2,
            infoType: Float = 0.2
        ) throws {
            self.aspectCount = aspectCount
            self.importance = importance
            self.knowledge = knowledge
            self.infoType = infoType
            
            guard self.isValid else {
                throw AspectAnalyzer.ComplexityError.invalidWeights("Weights must sum to 1.0")
            }
        }
    }
    
    /// Normalization thresholds for different factors
    public struct Thresholds: Sendable {
        /// Maximum number of aspects for normalization (default: 5)
        public let maxAspects: Int
        /// Maximum number of knowledge areas for normalization (default: 5)
        public let maxKnowledge: Int
        /// Maximum number of info types for normalization (default: 3)
        public let maxInfoTypes: Int
        
        public init(
            maxAspects: Int = 5,
            maxKnowledge: Int = 5,
            maxInfoTypes: Int = 3
        ) {
            self.maxAspects = maxAspects
            self.maxKnowledge = maxKnowledge
            self.maxInfoTypes = maxInfoTypes
        }
    }
    
    /// Weights for complexity calculation
    public let weights: Weights
    /// Thresholds for factor normalization
    public let thresholds: Thresholds
    
    public init(
        weights: Weights? = nil,
        thresholds: Thresholds? = nil
    ) throws {
        self.weights = try weights ?? Weights()
        self.thresholds = thresholds ?? Thresholds()
    }
    
    public static var `default`: ComplexityConfiguration {
        try! .init()
    }
}

extension AspectAnalyzer {
    /// Errors related to complexity calculation
    public enum ComplexityError: Error {
        case invalidWeights(String)
    }
    
    /// Calculates query complexity using provided configuration
    public func calculateComplexity(
        query: String,
        aspects: [Aspect],
        configuration: ComplexityConfiguration = .default
    ) -> Float {
        // 1. Aspect count factor
        let aspectCountFactor = Float(aspects.count) / Float(configuration.thresholds.maxAspects)
        let normalizedAspectCount = min(aspectCountFactor, 1.0)
        
        // 2. Average importance factor
        let avgImportance = aspects.isEmpty ? 0.0 :
        Float(aspects.map(\.importance).reduce(0.0, +)) / Float(aspects.count)
        
        // 3. Knowledge breadth factor
        let uniqueKnowledge = Set(aspects.flatMap(\.requiredKnowledge))
        let knowledgeFactor = Float(uniqueKnowledge.count) / Float(configuration.thresholds.maxKnowledge)
        let normalizedKnowledge = min(knowledgeFactor, 1.0)
        
        // 4. Information type diversity factor
        let uniqueInfoTypes = Set(aspects.flatMap(\.expectedInfoTypes))
        let infoTypeFactor = Float(uniqueInfoTypes.count) / Float(configuration.thresholds.maxInfoTypes)
        let normalizedInfoTypes = min(infoTypeFactor, 1.0)
        
        // Calculate weighted complexity
        let complexity =
        configuration.weights.aspectCount * normalizedAspectCount +
        configuration.weights.importance * avgImportance +
        configuration.weights.knowledge * normalizedKnowledge +
        configuration.weights.infoType * normalizedInfoTypes
        
        return min(max(complexity, 0.0), 1.0)
    }
    
    /// Detailed complexity analysis result
    public struct ComplexityAnalysis: Sendable {
        /// Overall complexity score
        public let score: Float
        /// Individual factor contributions
        public let factors: Factors
        
        public struct Factors: Sendable {
            /// Normalized aspect count contribution
            public let aspectCount: Float
            /// Average importance contribution
            public let importance: Float
            /// Normalized knowledge breadth contribution
            public let knowledge: Float
            /// Normalized information type diversity contribution
            public let infoTypes: Float
            
            /// Raw counts before normalization
            public let rawCounts: RawCounts
        }
        
        public struct RawCounts: Sendable {
            public let aspectCount: Int
            public let knowledgeAreas: Int
            public let infoTypes: Int
        }
    }
    
    /// Calculates detailed complexity analysis
    public func analyzeComplexity(
        query: String,
        aspects: [Aspect],
        config: ComplexityConfiguration
    ) -> ComplexityAnalysis {
        // Calculate raw factors
        let aspectCount = aspects.count
        let uniqueKnowledge = Set(aspects.flatMap(\.requiredKnowledge))
        let uniqueInfoTypes = Set(aspects.flatMap(\.expectedInfoTypes))
        
        // Calculate normalized factors
        let normalizedAspectCount = min(Float(aspectCount) / Float(config.thresholds.maxAspects), 1.0)
        let avgImportance = aspects.isEmpty ? 0.0 :
        Float(aspects.map(\.importance).reduce(0.0, +)) / Float(aspects.count)
        let normalizedKnowledge = min(Float(uniqueKnowledge.count) / Float(config.thresholds.maxKnowledge), 1.0)
        let normalizedInfoTypes = min(Float(uniqueInfoTypes.count) / Float(config.thresholds.maxInfoTypes), 1.0)
        
        // Calculate weighted contributions
        let aspectContribution = config.weights.aspectCount * normalizedAspectCount
        let importanceContribution = config.weights.importance * avgImportance
        let knowledgeContribution = config.weights.knowledge * normalizedKnowledge
        let infoTypeContribution = config.weights.infoType * normalizedInfoTypes
        
        let score = min(max(
            aspectContribution +
            importanceContribution +
            knowledgeContribution +
            infoTypeContribution,
            0.0
        ), 1.0)
        
        return ComplexityAnalysis(
            score: score,
            factors: .init(
                aspectCount: aspectContribution,
                importance: importanceContribution,
                knowledge: knowledgeContribution,
                infoTypes: infoTypeContribution,
                rawCounts: .init(
                    aspectCount: aspectCount,
                    knowledgeAreas: uniqueKnowledge.count,
                    infoTypes: uniqueInfoTypes.count
                )
            )
        )
    }
}
