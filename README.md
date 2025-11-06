# MLIR Hardware Accelerator Compiler - Learning Roadmap

This repository contains a comprehensive learning path for building a minimal MLIR-based compiler for hardware accelerators.

## 📚 Roadmaps Available

### 🚀 **Main: 4-Week Intensive** (`MLIR_COMPILER_ROADMAP.md`)
**Timeline**: 4 weeks (~20 hours/week = 80 hours total)

A focused, achievable roadmap that builds a complete minimal compiler in 4 weeks:
- **Week 1**: Foundation & minimal neural network dialect
- **Week 2**: Pattern rewriting & loop tiling optimization
- **Week 3**: Tensor-to-memory lowering & end-to-end pipeline
- **Week 4**: Testing, validation & final project

**Target**: A working compiler that optimizes a 2-layer MLP with loop tiling for cache efficiency.

**Best for**:
- Time-constrained learners (4-week deadline)
- Those who want a complete working project quickly
- Focus on core concepts over comprehensive coverage

### 📖 **Alternative: Comprehensive** (`MLIR_COMPILER_ROADMAP_COMPREHENSIVE.md`)
**Timeline**: 9-13 weeks (~15-20 hours/week)

A deeper, more comprehensive roadmap covering:
- More detailed explanations of concepts
- Additional optimization techniques
- Broader dialect design
- More extensive hardware optimization patterns

**Best for**:
- Those with more time to learn
- Wanting comprehensive understanding
- Planning to contribute to production compilers

## 🎯 Getting Started

### Prerequisites
- C++ programming knowledge
- Basic understanding of neural network architectures
- Linux/macOS environment (or WSL on Windows)

### Quick Start

1. **Clone this repository**
   ```bash
   git clone <your-repo-url>
   cd mlir
   ```

2. **Start with Phase 0: Hello World** (Day 1-2)
   ```bash
   cd mlir-hello-world
   # Follow instructions in README.md
   ```

3. **Follow the roadmap**
   - Open `MLIR_COMPILER_ROADMAP.md`
   - Follow the weekly structure
   - Complete exercises and deliverables
   - Track your progress

## 📂 Repository Structure

```
mlir/
├── README.md                                  # This file
├── MLIR_COMPILER_ROADMAP.md                   # Main 4-week roadmap
├── MLIR_COMPILER_ROADMAP_COMPREHENSIVE.md     # Extended version
└── mlir-hello-world/                          # Phase 0: Hello World project
    ├── main.cpp
    ├── CMakeLists.txt
    └── README.md
```

## 🎓 What You'll Learn

By completing the 4-week roadmap, you will:

✅ Understand MLIR fundamentals (operations, types, dialects)
✅ Build a custom neural network dialect
✅ Implement pattern-based optimizations
✅ Apply hardware-oriented optimizations (loop tiling)
✅ Lower from high-level tensors to low-level memory operations
✅ Create end-to-end compilation pipelines
✅ Validate compiler correctness and performance

## 🛠️ Technologies

- **MLIR**: Multi-Level Intermediate Representation
- **LLVM**: Compiler infrastructure
- **TableGen**: Code generation tool
- **C++17**: Primary implementation language
- **CMake & Ninja**: Build system

## 📊 Expected Outcome

A complete compiler that:
- Takes a 2-layer neural network as input
- Applies operation fusion (dense + relu)
- Tiles matrix multiplications for cache efficiency
- Generates optimized executable code
- Achieves 2-5x speedup from optimizations

## 🤝 Getting Help

- **MLIR Documentation**: https://mlir.llvm.org/
- **MLIR Discourse**: https://discourse.llvm.org/c/mlir/31
- **LLVM Discord**: https://discord.gg/xS7Z362

## 📝 License

This educational material is provided as-is for learning purposes.

## 🚀 Let's Build!

Start with the Hello World project and work through the roadmap at your own pace. Remember: the goal is to understand the fundamentals and build something that works, not to create a production compiler in 4 weeks.

Good luck on your MLIR journey! 🎉
