# AspectAnalyzer

AspectAnalyzer is a Swift library that performs semantic analysis of queries using Large Language Models. It breaks down queries into key aspects, evaluates their importance, and calculates query complexity.

## Features

- 🔍 **Semantic Query Analysis**: Breaks down complex queries into distinct aspects
- 📊 **Importance Evaluation**: Assigns importance scores to each identified aspect
- 🧩 **Knowledge Area Mapping**: Identifies required knowledge domains
- 📈 **Complexity Calculation**: Computes query complexity based on multiple factors
- 🎯 **Focus Area Detection**: Determines primary focus areas of queries

## Installation

### Swift Package Manager

Add the following to your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/1amageek/AspectAnalyzer.git", branch: "main")
]
```

## Quick Start

```swift
import AspectAnalyzer

// Initialize analyzer
let analyzer = AspectAnalyzer(model: "llama3.2")

// Analyze a query
let analysis = try await analyzer.analyzeQuery(
    "What is the impact of quantum computing on cryptography?"
)

// Access analysis results
print("Complexity score:", analysis.complexityScore)
print("\nPriority aspects:")
for aspect in analysis.prioritizedAspects {
    print("- \(aspect.description) (importance: \(aspect.importance))")
    print("  Knowledge areas: \(aspect.requiredKnowledge.joined(separator: ", "))")
    print("  Info types: \(aspect.expectedInfoTypes.joined(separator: ", "))")
}
```

## Detailed Usage

### Configuring Complexity Calculation

You can customize how complexity is calculated by providing a `ComplexityConfig`:

```swift
// Create custom configuration
let config = try ComplexityConfig(
    weights: .init(
        aspectCount: 0.4,  // Weight for number of aspects
        importance: 0.3,   // Weight for importance scores
        knowledge: 0.2,    // Weight for knowledge breadth
        infoType: 0.1     // Weight for information type diversity
    ),
    thresholds: .init(
        maxAspects: 7,     // Maximum aspects for normalization
        maxKnowledge: 5,   // Maximum knowledge areas
        maxInfoTypes: 4    // Maximum information types
    )
)

// Analyze with custom config
let analysis = try await analyzer.analyzeQuery(
    "Your query here",
    config: config
)
```

### Detailed Complexity Analysis

Get detailed insights into complexity calculations:

```swift
let analysis = try await analyzer.analyzeComplexity(
    query: "Your query",
    aspects: aspects,
    config: config
)

print("Overall complexity:", analysis.score)
print("Factor contributions:")
print("- Aspect count:", analysis.factors.aspectCount)
print("- Importance:", analysis.factors.importance)
print("- Knowledge breadth:", analysis.factors.knowledge)
print("- Info type diversity:", analysis.factors.infoTypes)
```

## Requirements

- Swift 5.9+
- macOS 15.0+ / iOS 18.0+
- Ollama (for LLM support)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This library is released under the MIT license. See [LICENSE](LICENSE) for details.

## Dependencies

- [OllamaKit](https://github.com/1amageek/OllamaKit): Swift interface for Ollama API
- [Logging](https://github.com/apple/swift-log): Apple's Swift Logging API

## Author

[@1amageek](https://x.com/1amageek)

## Acknowledgments

- [Ollama](https://ollama.com) for providing the language model capabilities
