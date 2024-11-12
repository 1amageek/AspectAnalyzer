// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "AspectAnalyzer",
    platforms: [.iOS(.v18), .macOS(.v15)],
    products: [
        .library(
            name: "AspectAnalyzer",
            targets: ["AspectAnalyzer"]),
    ],
    dependencies: [
        .package(url: "https://github.com/1amageek/SwiftRetry.git", branch: "main"),
        .package(url: "https://github.com/1amageek/OllamaKit.git", branch: "main"),
        .package(url: "https://github.com/apple/swift-log.git", branch: "main"),
    ],
    targets: [
        .target(
            name: "AspectAnalyzer",
            dependencies: [
                "SwiftRetry",
                "OllamaKit",
                .product(name: "Logging", package: "swift-log")
            ]
        ),
        .testTarget(
            name: "AspectAnalyzerTests",
            dependencies: ["AspectAnalyzer"]
        ),
    ]
)
