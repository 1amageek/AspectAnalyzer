import Testing
@testable import AspectAnalyzer
import Logging

@Test
func example() async throws {
    let analyzer = AspectAnalyzer(
        logger: Logger(label: "QueryAnalyzer")
    )
    
    let query = """
        Explain the impact of quantum computing on modern cryptography, \
        including potential vulnerabilities in current encryption methods \
        and proposed quantum-resistant alternatives.
        """
    
    let analysis = try await analyzer.analyzeQuery(query)
    
    // Print overall analysis
    print("Query Analysis Results")
    print("=====================")
    print("Query complexity: \(String(format: "%.2f", analysis.complexityScore))")
    
    // Print prioritized aspects with keywords
    print("\nPriority Aspects:")
    print("----------------")
    for aspect in analysis.prioritizedAspects {
        print("- \(aspect.description) (importance: \(String(format: "%.2f", aspect.importance)))")
        print("  Knowledge areas: \(aspect.requiredKnowledge.joined(separator: ", "))")
        print("  Info types: \(aspect.expectedInfoTypes.joined(separator: ", "))")
        
        // Print keywords by category
        if !aspect.keywords.isEmpty {
            print("  Keywords by category:")
            for (category, keywords) in aspect.keywordsByCategory {
                print("    \(category):")
                for keyword in keywords.sorted(by: { $0.weight > $1.weight }) {
                    print("      - \(keyword.term) (weight: \(String(format: "%.2f", keyword.weight)))")
                    if let context = keyword.context {
                        print("        context: \(context)")
                    }
                }
            }
        }
        
        // Print critical keywords
        let criticalKeywords = aspect.criticalKeywords
        if !criticalKeywords.isEmpty {
            print("  Critical Keywords:")
            for keyword in criticalKeywords {
                print("    - \(keyword.term) (weight: \(String(format: "%.2f", keyword.weight)))")
            }
        }
        print("")
    }
    
    // Print critical aspects summary
    print("\nCritical Aspects:")
    print("----------------")
    for aspect in analysis.criticalAspects {
        print("- \(aspect.description)")
        if !aspect.criticalKeywords.isEmpty {
            print("  Critical terms:")
            for keyword in aspect.criticalKeywords {
                print("    - \(keyword.term)")
            }
        }
    }
    
    // Print primary focus areas
    print("\nPrimary Focus Areas:")
    print("------------------")
    for area in analysis.primaryFocus {
        print("- \(area)")
    }
    
    // Print keyword analysis summary
    print("\nKeyword Analysis Summary:")
    print("-----------------------")
    let allKeywords = analysis.aspects.flatMap(\.keywords)
    let uniqueCategories = Set(allKeywords.map(\.category))
    
    for category in uniqueCategories.sorted() {
        let categoryKeywords = allKeywords.filter { $0.category == category }
        let avgWeight = categoryKeywords.map(\.weight).reduce(0, +) / Float(categoryKeywords.count)
        print("Category: \(category)")
        print("  Count: \(categoryKeywords.count)")
        print("  Average weight: \(String(format: "%.2f", avgWeight))")
        print("  Top terms: \(categoryKeywords.sorted { $0.weight > $1.weight }.prefix(3).map(\.term).joined(separator: ", "))")
        print("")
    }
}

@Test
func simpleKeywordExample() async throws {
    let analyzer = AspectAnalyzer(
        logger: Logger(label: "QueryAnalyzer")
    )
    
    let queries = [
        "What are the key benefits of using microservices architecture?",
        "How does climate change affect biodiversity in rainforests?",
        "Explain the basics of neural networks in machine learning."
    ]
    
    print("Quick Keyword Analysis Examples")
    print("==============================")
    
    for query in queries {
        let analysis = try await analyzer.extractKeywords(query)
        print("\nQuery:", analysis.query)
        print("Keywords:", analysis.keywords.joined(separator: ", "))
    }
}

@Test
func sampleAnalysisExamples() async throws {
    // Create analyzer instance with logging
    let analyzer = AspectAnalyzer(
        logger: Logger(label: "QueryAnalyzer")
    )
    
    // Sample queries covering different domains
    let queries = [
        "What are the key benefits of using microservices architecture?",
        "How does climate change affect biodiversity in rainforests?",
        "Explain the basics of neural networks in machine learning."
    ]
    
    // Keyword Analysis Examples
    print("\nQuick Description Analysis Examples")
    print("==============================")
    
    for query in queries {
        let analysis = try await analyzer.extractDescription(query)
        print("\nQuery:", analysis.query)
        print("Description:", analysis.description)
    }
}

@Test("Analyzes quantum computing cryptography query correctly")
func testQuantumCryptographyQuery() async throws {
    let analyzer = AspectAnalyzer(
        logger: Logger(label: "TestAnalyzer")
    )
    
    let query = """
    Explain the impact of quantum computing on modern cryptography, \
    including potential vulnerabilities in current encryption methods \
    and proposed quantum-resistant alternatives.
    """
    
    let analysis = try await analyzer.analyzeQuery(query)
    
    // Test overall structure
    #expect(!analysis.aspects.isEmpty, "Analysis should contain aspects")
    #expect(analysis.complexityScore > 0, "Complexity score should be greater than 0")
    #expect(!analysis.primaryFocus.isEmpty, "Should identify primary focus areas")
    
    // Test aspects content
    let aspects = analysis.aspects
    #expect(aspects.contains { $0.description.contains("quantum") }, "Should contain quantum computing aspect")
    #expect(aspects.contains { $0.description.contains("cryptography") || $0.description.contains("encryption") },
            "Should contain cryptography/encryption aspect")
    
    // Test knowledge areas
    let allKnowledge = Set(aspects.flatMap(\.requiredKnowledge))
    #expect(allKnowledge.contains { $0.contains("quantum") || $0.contains("cryptography") },
            "Should require quantum computing or cryptography knowledge")
    
    // Test information types
    let allInfoTypes = Set(aspects.flatMap(\.expectedInfoTypes))
    #expect(allInfoTypes.contains { $0.contains("technical") || $0.contains("theoretical") },
            "Should expect technical or theoretical information")
    
    // Test prioritization
    let prioritized = analysis.prioritizedAspects
    #expect(prioritized.first?.importance ?? 0 >= prioritized.last?.importance ?? 0,
            "Aspects should be correctly prioritized")
    
    // Test critical aspects
    let critical = analysis.criticalAspects
    #expect(!critical.isEmpty, "Should identify critical aspects")
    #expect(critical.allSatisfy { $0.importance > 0.7 }, "Critical aspects should have high importance")
}

@Test("Analyzes simple query with appropriate complexity")
func testSimpleQuery() async throws {
    let analyzer = AspectAnalyzer(
        logger: Logger(label: "TestAnalyzer")
    )
    
    let query = "What is the weather like today?"
    let analysis = try await analyzer.analyzeQuery(query)
    
    // Simple queries should have lower complexity
    #expect(analysis.complexityScore < 0.5,
            "Simple query should have low complexity score")
    
    // Should have fewer aspects
    #expect(analysis.aspects.count <= 3,
            "Simple query should have few aspects")
    
    // Should have focused knowledge areas
    let knowledgeAreas = Set(analysis.aspects.flatMap(\.requiredKnowledge))
    #expect(knowledgeAreas.count <= 2,
            "Simple query should have focused knowledge areas")
}

@Test("Analyzes complex technical query with high complexity")
func testComplexTechnicalQuery() async throws {
    let analyzer = AspectAnalyzer(
        logger: Logger(label: "TestAnalyzer")
    )
    
    let query = """
    Compare and contrast different deep learning architectures for natural \
    language processing, focusing on transformer models, attention mechanisms, \
    and their applications in multi-modal learning scenarios.
    """
    
    let analysis = try await analyzer.analyzeQuery(query)
    
    // Complex queries should have higher complexity
    #expect(analysis.complexityScore > 0.7,
            "Complex technical query should have high complexity score")
    
    // Should have multiple aspects
    #expect(analysis.aspects.count >= 3,
            "Complex query should have multiple aspects")
    
    // Should have diverse knowledge areas
    let knowledgeAreas = Set(analysis.aspects.flatMap(\.requiredKnowledge))
    #expect(knowledgeAreas.count >= 3,
            "Complex query should have diverse knowledge areas")
    
    // Should have multiple critical aspects
    #expect(analysis.criticalAspects.count >= 1,
            "Complex query should have multiple critical aspects")
}



@Test("Primary focus areas are correctly identified")
func testPrimaryFocusAreas() async throws {
    let analyzer = AspectAnalyzer(
        logger: Logger(label: "TestAnalyzer")
    )
    
    let query = """
    Discuss the environmental impact of electric vehicles, including battery \
    production, charging infrastructure, and comparison with traditional vehicles.
    """
    
    let analysis = try await analyzer.analyzeQuery(query)
    
    // Test primary focus identification
    let focus = analysis.primaryFocus
    #expect(focus.contains { $0.contains("environmental") || $0.contains("sustainability") },
            "Should identify environmental impact as primary focus")
    #expect(focus.contains { $0.contains("electric") || $0.contains("automotive") },
            "Should identify electric vehicles as primary focus")
    
    // Test focus area prioritization
    let prioritizedAspects = analysis.prioritizedAspects
    let topAspect = try #require(prioritizedAspects.first)
    #expect(focus.contains { topAspect.requiredKnowledge.contains($0) },
            "Primary focus should align with highest priority aspect")
}

@Test("Complexity calculation is consistent")
func testComplexityCalculation() async throws {
    let analyzer = AspectAnalyzer(
        logger: Logger(label: "TestAnalyzer")
    )
    
    // Test multiple queries of varying complexity
    let queries = [
        "What time is it?",
        "Explain how photosynthesis works.",
        """
        Analyze the socioeconomic factors contributing to climate change, \
        including industrial policies, consumer behavior, and international \
        cooperation frameworks.
        """
    ]
    
    var complexities: [Float] = []
    
    for query in queries {
        let analysis = try await analyzer.analyzeQuery(query)
        complexities.append(analysis.complexityScore)
    }
    
    // Test that complexity increases with query complexity
    #expect(complexities[0] < complexities[1],
            "Simple query should have lower complexity than moderate query")
    #expect(complexities[1] < complexities[2],
            "Moderate query should have lower complexity than complex query")
    
    // Test complexity bounds
    #expect(complexities.allSatisfy { $0 >= 0 && $0 <= 1 },
            "Complexity scores should be between 0 and 1")
}
