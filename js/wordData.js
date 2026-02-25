// 大模型术语学习数据
const KnowledgeItems = [
  {
    id: 1,
    root: "Transformer",
    origin: "Architecture",
    meaning: "注意力机制架构",
    description: "Transformer是由Google在2017年《Attention is All You Need》论文中提出的深度学习架构。它摒弃了传统的RNN和CNN，完全基于注意力机制（Attention Mechanism）处理序列数据。Transformer的核心创新是自注意力（Self-Attention）机制，能够并行处理所有位置的信息，极大提升了训练效率。现在几乎所有主流大模型（GPT、BERT、Claude、Gemini）都基于Transformer架构。",
    examples: [
      {
        word: "GPT系列",
        meaning: "生成式预训练模型",
        breakdown: { root: "Transformer" },
        explanation: "GPT（Generative Pre-trained Transformer）使用Transformer的解码器（Decoder）部分，通过自回归方式生成文本。从GPT-1到GPT-4，模型规模不断扩大，能力持续提升。"
      },
      {
        word: "BERT",
        meaning: "双向编码器表示",
        breakdown: { root: "Transformer" },
        explanation: "BERT（Bidirectional Encoder Representations from Transformers）使用Transformer的编码器（Encoder）部分，通过双向注意力机制理解上下文，在问答、分类等任务上表现优异。"
      },
      {
        word: "T5",
        meaning: "文本到文本转换",
        breakdown: { root: "Transformer" },
        explanation: "T5（Text-to-Text Transfer Transformer）将所有NLP任务统一为文本到文本的转换问题，同时使用编码器和解码器，实现了任务无关的通用架构。"
      }
    ],
    quiz: {
      question: "Transformer的核心创新是什么？",
      options: [
        "循环神经网络（RNN）",
        "自注意力机制（Self-Attention）",
        "卷积神经网络（CNN）",
        "全连接层（Fully Connected）"
      ],
      correctAnswer: 1
    }
  },
  {
    id: 2,
    root: "Attention",
    origin: "Mechanism",
    meaning: "注意力机制",
    description: "Attention机制是深度学习中的一种技术，让模型能够「关注」输入的重要部分。就像人类阅读时会重点关注关键词一样，注意力机制通过计算Query（查询）、Key（键）、Value（值）之间的相关性，为不同位置分配不同的权重。Self-Attention是特殊的注意力机制，让序列中的每个元素都能关注序列中的所有其他元素，捕捉长距离依赖关系。",
    examples: [
      {
        word: "Multi-Head Attention",
        meaning: "多头注意力",
        breakdown: { root: "Attention" },
        explanation: "将注意力机制分成多个「头」（Head），每个头学习不同的特征模式。比如一个头关注语法，另一个头关注语义。最后将所有头的输出拼接起来，丰富表示能力。"
      },
      {
        word: "Cross-Attention",
        meaning: "交叉注意力",
        breakdown: { root: "Attention" },
        explanation: "让一个序列（如翻译的目标语言）关注另一个序列（如源语言）。在编码器-解码器架构中，解码器通过Cross-Attention从编码器获取信息。"
      },
      {
        word: "Scaled Dot-Product",
        meaning: "缩放点积注意力",
        breakdown: { root: "Attention" },
        explanation: "Transformer使用的标准注意力计算方法：Attention(Q,K,V) = softmax(QK^T/√d)V。除以√d是为了防止点积结果过大导致梯度消失。"
      }
    ],
    quiz: {
      question: "Self-Attention的主要作用是什么？",
      options: [
        "加快训练速度",
        "减少模型参数",
        "捕捉序列内的长距离依赖",
        "降低计算复杂度"
      ],
      correctAnswer: 2
    }
  },
  {
    id: 3,
    root: "Token",
    origin: "Concept",
    meaning: "词元/标记",
    description: "Token是大模型处理文本的基本单位。由于神经网络无法直接理解文字，需要将文本切分成更小的片段（Token），然后转换成数字表示（Token ID）输入模型。一个Token可能是一个完整的词（如'apple'）、词的一部分（如'un-'、'-ing'）、单个字符、甚至标点符号。英文平均1个Token约等于0.75个单词，中文平均1个Token约等于1.5-2个汉字。",
    examples: [
      {
        word: "Tokenization",
        meaning: "分词/标记化",
        breakdown: { root: "Token" },
        explanation: "将文本切分成Token序列的过程。常用算法有BPE（Byte Pair Encoding）、WordPiece、SentencePiece。GPT使用BPE，BERT使用WordPiece。"
      },
      {
        word: "Token Limit",
        meaning: "Token限制",
        breakdown: { root: "Token" },
        explanation: "大模型的上下文窗口限制，如GPT-3.5是4K tokens，GPT-4是8K/32K，Claude 3是200K。超过限制就无法处理，需要截断或分段。"
      },
      {
        word: "Subword Token",
        meaning: "子词标记",
        breakdown: { root: "Token" },
        explanation: "介于字符和单词之间的Token粒度。比如'unhappiness'可能被切分为'un-'、'happiness'，既能处理生僻词，又比字符级更高效。"
      }
    ],
    quiz: {
      question: "为什么大模型要使用Token而不是直接处理单词？",
      options: [
        "Token更容易理解",
        "可以处理未见过的词和多语言",
        "Token占用空间更小",
        "计算速度更快"
      ],
      correctAnswer: 1
    }
  },
  {
    id: 4,
    root: "Embedding",
    origin: "Representation",
    meaning: "嵌入/向量表示",
    description: "Embedding是将Token（文字）转换为高维向量（数字）的过程。比如将'cat'转换为[0.2, -0.5, 0.8, ...]这样的向量，向量中的每个维度代表某种语义特征。相似的词在向量空间中距离较近，如'cat'和'dog'的向量会比'cat'和'car'更接近。大模型通常使用几百到几千维的Embedding，通过训练让这些向量蕴含丰富的语义信息。",
    examples: [
      {
        word: "Word2Vec",
        meaning: "词向量",
        breakdown: { root: "Embedding" },
        explanation: "Google在2013年提出的经典词嵌入方法，通过浅层神经网络学习词向量。著名例子：king - man + woman ≈ queen，展示了向量运算可以捕捉语义关系。"
      },
      {
        word: "Position Embedding",
        meaning: "位置嵌入",
        breakdown: { root: "Embedding" },
        explanation: "由于Transformer没有内置位置信息，需要额外的位置编码告诉模型词的先后顺序。可以是固定的三角函数编码，也可以是可学习的参数。"
      },
      {
        word: "Contextual Embedding",
        meaning: "上下文嵌入",
        breakdown: { root: "Embedding" },
        explanation: "同一个词在不同上下文中有不同的向量表示。比如'bank'在'river bank'和'savings bank'中的Embedding不同，BERT等模型就是生成上下文相关的Embedding。"
      }
    ],
    quiz: {
      question: "Embedding的主要作用是什么？",
      options: [
        "压缩文本大小",
        "将文字转换为可计算的向量",
        "加密文本内容",
        "生成文本摘要"
      ],
      correctAnswer: 1
    }
  },
  {
    id: 5,
    root: "Fine-tuning",
    origin: "Training",
    meaning: "微调",
    description: "Fine-tuning是在预训练模型基础上，用特定任务的数据继续训练，让模型适应新任务的过程。就像一个通才（预训练模型）经过专业培训后成为专家。相比从头训练，微调只需要少量数据和计算资源，就能获得很好的效果。现在大部分应用都是先用通用大模型（如GPT-4）预训练，再针对具体场景（如客服、医疗、法律）微调。",
    examples: [
      {
        word: "Full Fine-tuning",
        meaning: "全量微调",
        breakdown: { root: "Fine-tuning" },
        explanation: "更新模型所有参数。效果最好但成本最高，需要大量显存。对于70B参数的模型，全量微调需要数百GB显存，普通硬件无法承受。"
      },
      {
        word: "LoRA",
        meaning: "低秩适配",
        breakdown: { prefix: "Lo", root: "RA" },
        explanation: "LoRA（Low-Rank Adaptation）只训练少量新增参数（通常<1%原模型参数），大幅降低显存需求。原理是在原模型旁边加入低秩矩阵，只训练这些矩阵。"
      },
      {
        word: "Instruction Tuning",
        meaning: "指令微调",
        breakdown: { root: "Fine-tuning" },
        explanation: "用「指令-回答」格式的数据微调模型，让模型学会遵循人类指令。如ChatGPT就是GPT-3.5经过指令微调得到的，大幅提升了对话能力。"
      }
    ],
    quiz: {
      question: "LoRA相比全量微调的主要优势是什么？",
      options: [
        "训练速度更快",
        "显存需求大幅降低",
        "效果更好",
        "不需要训练数据"
      ],
      correctAnswer: 1
    }
  },
  {
    id: 6,
    root: "Prompt",
    origin: "Input",
    meaning: "提示词",
    description: "Prompt是用户输入给大模型的文本指令，相当于跟AI对话时说的话。好的Prompt能极大提升模型输出质量。Prompt Engineering（提示词工程）是一门新兴技能，研究如何设计有效的Prompt。常见技巧包括：给出明确指令、提供示例（Few-shot）、分步骤引导（Chain of Thought）、设定角色（如'你是一个专业的翻译'）。",
    examples: [
      {
        word: "Zero-shot",
        meaning: "零样本提示",
        breakdown: { root: "Prompt" },
        explanation: "不提供任何示例，直接让模型完成任务。如'将以下文本翻译成英文：...'。依赖模型预训练时学到的知识。"
      },
      {
        word: "Few-shot",
        meaning: "少样本提示",
        breakdown: { root: "Prompt" },
        explanation: "在Prompt中给出几个示例，让模型理解任务模式。如给3个「问题-答案」示例，然后让模型回答新问题。通常比Zero-shot效果更好。"
      },
      {
        word: "Chain of Thought",
        meaning: "思维链",
        breakdown: { root: "Prompt" },
        explanation: "要求模型「一步步思考」，输出推理过程而不是直接给答案。能显著提升复杂推理任务的准确率，特别是数学和逻辑问题。"
      }
    ],
    quiz: {
      question: "Few-shot Prompting的特点是什么？",
      options: [
        "不需要提供示例",
        "在Prompt中包含少量示例",
        "需要大量训练数据",
        "只能用于分类任务"
      ],
      correctAnswer: 1
    }
  },
  {
    id: 7,
    root: "Hallucination",
    origin: "Problem",
    meaning: "幻觉",
    description: "Hallucination指大模型生成看似合理但实际错误或无中生有的内容。比如捏造不存在的论文引用、编造虚假统计数据、给出错误的历史事件。这是当前大模型最严重的问题之一。原因在于：模型本质是根据概率生成文本，没有真正的「知识」和「事实核查」能力；训练数据中的错误会被学习；模型倾向于生成「听起来像答案」的内容。缓解方法包括检索增强生成（RAG）、加强事实性训练、要求模型承认不确定性。",
    examples: [
      {
        word: "Factual Hallucination",
        meaning: "事实性幻觉",
        breakdown: { root: "Hallucination" },
        explanation: "生成与事实不符的内容。如说'埃菲尔铁塔在伦敦'、'iPhone 15有8个摄像头'。这类错误尤其危险，因为模型表述很自信，容易误导用户。"
      },
      {
        word: "Intrinsic Hallucination",
        meaning: "内在幻觉",
        breakdown: { root: "Hallucination" },
        explanation: "生成与输入内容矛盾的回答。如文档明确说'产品价格是99元'，模型却回答'该产品价格是129元'。在摘要、问答任务中常见。"
      },
      {
        word: "Extrinsic Hallucination",
        meaning: "外在幻觉",
        breakdown: { root: "Hallucination" },
        explanation: "生成无法从输入验证但实际错误的内容。如基于一篇关于AI的文章，模型添加了文章中没有的'作者获得图灵奖'等信息。"
      }
    ],
    quiz: {
      question: "如何缓解大模型的幻觉问题？",
      options: [
        "增加模型参数量",
        "使用检索增强生成（RAG）",
        "加快推理速度",
        "降低Temperature参数"
      ],
      correctAnswer: 1
    }
  },
  {
    id: 8,
    root: "Temperature",
    origin: "Parameter",
    meaning: "温度参数",
    description: "Temperature控制模型输出的随机性和创造性。取值范围通常是0-2，默认1。Temperature=0时，模型总是选择概率最高的词，输出确定性强、保守；Temperature越高，模型越可能选择概率较低的词，输出更有创意但也更不可预测。简单理解：Temperature低=严谨、一致；Temperature高=创意、多样。写代码、翻译用低温度（0.2-0.5），创意写作、头脑风暴用高温度（0.7-1.5）。",
    examples: [
      {
        word: "Greedy Decoding",
        meaning: "贪心解码",
        breakdown: { root: "Temperature" },
        explanation: "Temperature=0的极端情况，每次都选择概率最高的Token。优点是输出稳定可复现，缺点是缺乏多样性，可能陷入重复。"
      },
      {
        word: "Top-k Sampling",
        meaning: "Top-k采样",
        breakdown: { root: "Temperature" },
        explanation: "只考虑概率最高的k个Token进行采样。如k=50，就只在最可能的50个词中随机选择。既保证质量，又有一定随机性。"
      },
      {
        word: "Nucleus Sampling",
        meaning: "核采样",
        breakdown: { root: "Temperature" },
        explanation: "也叫Top-p采样，选择累积概率达到p的最小Token集合。如p=0.9，就选择概率总和为90%的最少Token数。比Top-k更灵活，根据概率分布自适应调整候选数量。"
      }
    ],
    quiz: {
      question: "什么任务适合使用较高的Temperature？",
      options: [
        "数学题求解",
        "代码生成",
        "创意写作",
        "文档翻译"
      ],
      correctAnswer: 2
    }
  },
  {
    id: 9,
    root: "Context Window",
    origin: "Limitation",
    meaning: "上下文窗口",
    description: "Context Window是大模型一次能处理的最大Token数量，包括输入和输出。可以理解为模型的「短期记忆容量」。如GPT-3.5是4K tokens（约3000字），GPT-4是8K/32K/128K，Claude 3是200K，Gemini 1.5 Pro甚至达到1M（约70万字）。超出窗口的内容会被遗忘或截断。长上下文带来巨大优势：可以处理整本书、完整代码库、长对话历史，但也带来计算成本急剧上升的挑战。",
    examples: [
      {
        word: "Sliding Window",
        meaning: "滑动窗口",
        breakdown: { root: "Context Window" },
        explanation: "处理超长文本的策略：保留最近的N个Token，丢弃更早的内容。如窗口4K，对话超过4K后，最早的内容会被「遗忘」。ChatGPT就是这样处理长对话的。"
      },
      {
        word: "RoPE",
        meaning: "旋转位置编码",
        breakdown: { root: "Context Window" },
        explanation: "Rotary Position Embedding，一种位置编码方法，能更好地支持长上下文。通过旋转操作编码位置信息，外推性能优于传统方法，被LLaMA等模型采用。"
      },
      {
        word: "Context Compression",
        meaning: "上下文压缩",
        breakdown: { root: "Context Window" },
        explanation: "用摘要、关键信息提取等方法压缩上下文，在有限窗口内塞入更多信息。如把10万字文档压缩成5000字摘要，再输入模型。"
      }
    ],
    quiz: {
      question: "为什么更长的Context Window需要更多计算资源？",
      options: [
        "需要存储更多Token",
        "注意力机制的复杂度是O(n²)",
        "模型参数更多",
        "训练数据更大"
      ],
      correctAnswer: 1
    }
  },
  {
    id: 10,
    root: "RAG",
    origin: "Technique",
    meaning: "检索增强生成",
    description: "RAG（Retrieval-Augmented Generation）是一种结合检索系统和生成模型的技术。工作流程：1) 根据用户问题从知识库检索相关文档；2) 将检索到的文档和问题一起输入大模型；3) 模型基于检索内容生成回答。RAG解决了大模型的两大痛点：幻觉问题（有真实文档支撑）和知识更新（知识库可随时更新，无需重新训练）。现在很多AI应用（如企业知识库问答、文档助手）都基于RAG架构。",
    examples: [
      {
        word: "Vector Database",
        meaning: "向量数据库",
        breakdown: { root: "RAG" },
        explanation: "存储文档Embedding的专用数据库，如Pinecone、Weaviate、Milvus。支持快速相似度搜索，是RAG系统的核心组件。把文档转成向量存储，查询时也转成向量，找出最相似的文档。"
      },
      {
        word: "Dense Retrieval",
        meaning: "稠密检索",
        breakdown: { root: "RAG" },
        explanation: "基于Embedding的语义检索，能理解问题和文档的深层含义。比传统关键词匹配更智能，如'如何煮面'能匹配到'意大利面烹饪方法'。"
      },
      {
        word: "Hybrid Search",
        meaning: "混合搜索",
        breakdown: { root: "RAG" },
        explanation: "结合关键词搜索（BM25）和向量搜索（Dense Retrieval）的优势。关键词搜索擅长精确匹配，向量搜索擅长语义理解，两者互补效果更好。"
      }
    ],
    quiz: {
      question: "RAG相比直接使用大模型的主要优势是什么？",
      options: [
        "生成速度更快",
        "减少幻觉并支持知识更新",
        "模型参数更少",
        "不需要GPU"
      ],
      correctAnswer: 1
    }
  },
  {
    id: 11,
    root: "RLHF",
    origin: "Training",
    meaning: "人类反馈强化学习",
    description: "RLHF（Reinforcement Learning from Human Feedback）是让大模型更符合人类偏好的训练方法，ChatGPT的成功关键技术。流程：1) 预训练基础模型；2) 人类标注员对模型输出排序（哪个回答更好）；3) 训练奖励模型（Reward Model）学习人类偏好；4) 用强化学习（PPO算法）优化生成模型，让它获得更高奖励。通过RLHF，模型学会了有帮助、诚实、无害（HHH）的行为准则。",
    examples: [
      {
        word: "Reward Model",
        meaning: "奖励模型",
        breakdown: { root: "RLHF" },
        explanation: "一个专门的模型，输入是一段文本，输出是质量分数。通过人类偏好数据训练，学会判断回答的好坏。在RLHF中充当'裁判'角色，指导生成模型优化。"
      },
      {
        word: "PPO",
        meaning: "近端策略优化",
        breakdown: { root: "RLHF" },
        explanation: "Proximal Policy Optimization，一种稳定的强化学习算法。在RLHF中用来更新生成模型参数，让模型输出获得更高奖励，同时避免偏离原始模型太远。"
      },
      {
        word: "Constitutional AI",
        meaning: "宪法AI",
        breakdown: { root: "RLHF" },
        explanation: "Anthropic提出的改进RLHF的方法。通过预定义的'宪法'（一组规则）让模型自我批评和改进，减少对人类标注的依赖。Claude就是用这个方法训练的。"
      }
    ],
    quiz: {
      question: "RLHF中的Reward Model的作用是什么？",
      options: [
        "生成文本",
        "评估输出质量",
        "压缩模型",
        "加速推理"
      ],
      correctAnswer: 1
    }
  },
  {
    id: 12,
    root: "Inference",
    origin: "Deployment",
    meaning: "推理",
    description: "Inference是用训练好的模型生成输出的过程，也叫「推断」或「预测」。与训练相比，推理不需要计算梯度和反向传播，但对响应速度要求更高（用户不愿等待）。推理成本占大模型运营成本的大头：GPT-4一次推理可能花费几分钱，每天数百万用户使用，成本惊人。优化推理的技术包括：量化（降低精度）、蒸馏（压缩模型）、KV缓存（避免重复计算）、批处理（同时处理多个请求）。",
    examples: [
      {
        word: "Batching",
        meaning: "批处理",
        breakdown: { root: "Inference" },
        explanation: "同时处理多个请求，共享计算资源。如同时为100个用户生成回答，比100次单独推理效率高得多。但批次太大会增加每个用户的等待时间，需要权衡。"
      },
      {
        word: "KV Cache",
        meaning: "键值缓存",
        breakdown: { root: "Inference" },
        explanation: "缓存注意力机制中的Key和Value矩阵，避免重复计算。在生成长文本时，每生成一个新Token，不需要重新计算之前所有Token的KV，直接用缓存，大幅加速。"
      },
      {
        word: "Speculative Decoding",
        meaning: "推测解码",
        breakdown: { root: "Inference" },
        explanation: "用小模型快速生成多个候选Token，大模型批量验证。如果小模型猜对了，就节省了大模型的多次串行推理。能提速2-3倍，且输出完全一致。"
      }
    ],
    quiz: {
      question: "KV Cache的主要作用是什么？",
      options: [
        "减少内存占用",
        "避免重复计算加速生成",
        "提高输出质量",
        "降低训练成本"
      ],
      correctAnswer: 1
    }
  },
  {
    id: 13,
    root: "Quantization",
    origin: "Optimization",
    meaning: "量化",
    description: "Quantization是降低模型数值精度以减少内存和计算量的技术。原理：将32位浮点数（FP32）转换为更低精度格式，如16位（FP16）、8位整数（INT8）甚至4位（INT4）。例如，70B参数的模型，FP32需要280GB内存，INT8只需70GB，INT4仅需35GB，能在消费级显卡运行。权衡：精度下降会略微影响性能，但大多数情况下影响很小。常用方法包括后训练量化（PTQ）和量化感知训练（QAT）。",
    examples: [
      {
        word: "GPTQ",
        meaning: "GPT量化",
        breakdown: { root: "Quantization" },
        explanation: "一种高效的后训练量化方法，专为大语言模型设计。能将模型量化到4-bit，几乎不损失性能。如70B模型量化后可在24GB显卡运行。"
      },
      {
        word: "GGUF",
        meaning: "通用量化格式",
        breakdown: { root: "Quantization" },
        explanation: "一种量化模型的存储格式，支持多种量化级别（Q4_0, Q5_K等）。llama.cpp项目使用这种格式，让量化模型能在CPU和普通硬件上运行。"
      },
      {
        word: "Mixed Precision",
        meaning: "混合精度",
        breakdown: { root: "Quantization" },
        explanation: "对模型不同部分使用不同精度。如大部分用INT8，关键层用FP16。既降低资源需求，又保持性能。训练时也常用混合精度加速。"
      }
    ],
    quiz: {
      question: "量化的主要目的是什么？",
      options: [
        "提高模型准确率",
        "减少内存和计算资源需求",
        "加快训练速度",
        "增加模型参数"
      ],
      correctAnswer: 1
    }
  },
  {
    id: 14,
    root: "Distillation",
    origin: "Compression",
    meaning: "蒸馏",
    description: "Distillation（知识蒸馏）是将大模型（Teacher）的知识转移到小模型（Student）的技术。过程：Teacher生成「软标签」（概率分布而非硬标签），Student学习模仿这些分布。蒸馏后的小模型保留了大模型的大部分能力，但体积小、速度快、成本低。典型例子：GPT-3.5可能是GPT-4蒸馏得到的，DistilBERT是BERT的蒸馏版本（参数减半，速度翻倍，性能损失<5%）。适用于对延迟敏感或资源受限的场景。",
    examples: [
      {
        word: "Hard Labels",
        meaning: "硬标签",
        breakdown: { root: "Distillation" },
        explanation: "传统的0/1标签，如分类任务中'是猫'(1)或'不是猫'(0)。硬标签损失了很多信息，如模型对不同错误答案的信心差异。"
      },
      {
        word: "Soft Labels",
        meaning: "软标签",
        breakdown: { root: "Distillation" },
        explanation: "Teacher模型输出的概率分布，如[猫:0.9, 狗:0.08, 鸟:0.02]。软标签包含了Teacher的'不确定性'和类别间关系，Student学习它比学硬标签效果更好。"
      },
      {
        word: "Self-Distillation",
        meaning: "自蒸馏",
        breakdown: { root: "Distillation" },
        explanation: "让模型蒸馏自己：用模型当前版本作为Teacher，训练改进后的版本。或者把模型不同时刻的输出互相蒸馏。能持续提升性能。"
      }
    ],
    quiz: {
      question: "知识蒸馏的Student模型相比Teacher模型有什么特点？",
      options: [
        "参数更多",
        "速度更慢",
        "体积更小、速度更快",
        "性能一定更好"
      ],
      correctAnswer: 2
    }
  },
  {
    id: 15,
    root: "MoE",
    origin: "Architecture",
    meaning: "专家混合",
    description: "MoE（Mixture of Experts）是一种稀疏激活架构：模型包含多个专家网络，每次推理只激活部分专家，而非全部参数。路由器（Router）决定哪些专家处理当前输入。优势：可以用更少的计算获得更多参数的效果。如Mixtral 8x7B有56B参数，但每次只用12B，推理成本接近12B模型。Gemini 1.5、GPT-4传言也使用MoE。挑战：训练MoE很难，容易出现专家负载不均、路由学习困难等问题。",
    examples: [
      {
        word: "Router Network",
        meaning: "路由网络",
        breakdown: { root: "MoE" },
        explanation: "决定激活哪些专家的小型网络。输入一个Token，路由器输出每个专家的权重，选择Top-K个专家处理。路由器的训练是MoE的关键难点。"
      },
      {
        word: "Expert Capacity",
        meaning: "专家容量",
        breakdown: { root: "MoE" },
        explanation: "每个专家能处理的Token数量上限。如果某专家被分配的Token超过容量，多余的Token会被丢弃或分给其他专家，导致负载不均问题。"
      },
      {
        word: "Load Balancing",
        meaning: "负载均衡",
        breakdown: { root: "MoE" },
        explanation: "确保所有专家被均匀使用的技术。如果某些专家总是被选中，其他专家被忽略，模型就退化成少数专家，失去MoE的优势。需要专门的损失函数鼓励负载均衡。"
      }
    ],
    quiz: {
      question: "MoE架构的主要优势是什么？",
      options: [
        "训练更简单",
        "用更少计算获得更多参数的效果",
        "推理速度更快",
        "不需要GPU"
      ],
      correctAnswer: 1
    }
  },
  {
    id: 16,
    root: "Agent",
    origin: "Application",
    meaning: "智能体",
    description: "Agent是能自主感知环境、制定计划、执行动作的AI系统。相比被动回答的聊天机器人，Agent更主动：能调用工具（搜索、计算器、API）、分解复杂任务、记忆上下文、迭代优化。经典框架：ReAct（推理+行动循环）、AutoGPT（自动设定和完成目标）。应用场景：代码助手（理解需求→写代码→测试→修复）、客服机器人（查询订单→检索政策→解决问题）、科研助手（查文献→总结→提出假设）。",
    examples: [
      {
        word: "ReAct",
        meaning: "推理-行动框架",
        breakdown: { root: "Agent" },
        explanation: "Reason + Act的结合。每一步Agent先推理（我现在应该做什么），然后行动（调用工具），观察结果，再进入下一轮。如解数学题：推理'需要查x的值'→行动'搜索x定义'→观察'x=10'→推理'现在能计算了'。"
      },
      {
        word: "Tool Use",
        meaning: "工具使用",
        breakdown: { root: "Agent" },
        explanation: "让大模型调用外部工具的能力。如调用搜索引擎（获取最新信息）、计算器（精确计算）、代码解释器（执行代码）、API（查询数据库）。GPT-4的插件、Claude的工具使用都是这个概念。"
      },
      {
        word: "Memory",
        meaning: "记忆系统",
        breakdown: { root: "Agent" },
        explanation: "Agent的长期记忆能力，超越单次对话的上下文窗口。包括短期记忆（当前任务上下文）、长期记忆（历史交互）、工作记忆（中间结果）。如客服Agent记住客户之前的问题和偏好。"
      }
    ],
    quiz: {
      question: "Agent相比普通聊天机器人的核心区别是什么？",
      options: [
        "回答速度更快",
        "能主动规划和使用工具",
        "模型参数更多",
        "不会产生幻觉"
      ],
      correctAnswer: 1
    }
  },
  {
    id: 17,
    root: "Multimodal",
    origin: "Capability",
    meaning: "多模态",
    description: "Multimodal指大模型能处理多种类型的数据：文本、图像、音频、视频。早期大模型只能处理文本，现在的模型（GPT-4V、Gemini、Claude 3）能「看图」「听声音」。技术关键是将不同模态统一映射到同一向量空间，让模型理解它们的关系。如看到图片[猫]，模型知道它对应文本'a cat sitting on a sofa'。应用：图片问答、视频理解、OCR、医学影像分析、自动驾驶感知。",
    examples: [
      {
        word: "Vision Encoder",
        meaning: "视觉编码器",
        breakdown: { root: "Multimodal" },
        explanation: "将图片转换为向量的模块，通常基于CNN或Vision Transformer（ViT）。如CLIP的图像编码器，把任意图片编码成768维向量，再输入语言模型。"
      },
      {
        word: "CLIP",
        meaning: "对比语言-图像预训练",
        breakdown: { root: "Multimodal" },
        explanation: "OpenAI的多模态模型，通过对比学习让图片和文本在同一空间对齐。训练数据：4亿组图文对。如图片[狗]和文本'a dog'距离近，'a cat'距离远。GPT-4V的视觉能力基于CLIP。"
      },
      {
        word: "Cross-Modal Attention",
        meaning: "跨模态注意力",
        breakdown: { root: "Multimodal" },
        explanation: "让模型在不同模态间建立关联的机制。如看图回答问题时，文本Query关注图片的相关区域。问'猫在哪'，注意力集中在图片中猫的位置。"
      }
    ],
    quiz: {
      question: "多模态大模型的关键技术是什么？",
      options: [
        "增加参数量",
        "将不同模态映射到统一向量空间",
        "加快训练速度",
        "使用更多GPU"
      ],
      correctAnswer: 1
    }
  },
  {
    id: 18,
    root: "Zero-shot",
    origin: "Learning",
    meaning: "零样本学习",
    description: "Zero-shot指模型在没有见过任何该任务示例的情况下，直接完成新任务。得益于预训练学到的广泛知识，大模型展现出惊人的零样本能力。如GPT-3训练时从未做过翻译任务，但能翻译上百种语言对。原理：预训练时学到了任务的「元知识」，如语言结构、常识、推理模式。指令：清晰描述任务即可，如'将以下文本分类为正面或负面：...'。Zero-shot是大模型「智能涌现」的重要体现。",
    examples: [
      {
        word: "Emergent Ability",
        meaning: "涌现能力",
        breakdown: { root: "Zero-shot" },
        explanation: "当模型规模达到一定阈值后，突然出现训练时未专门学习的能力。如算术推理、逻辑推理、代码生成等能力，在小模型几乎不存在，大模型突然变得很强。"
      },
      {
        word: "In-Context Learning",
        meaning: "上下文学习",
        breakdown: { root: "Zero-shot" },
        explanation: "模型从Prompt中的示例学习任务模式，无需更新参数。Zero-shot是0个示例，Few-shot是几个示例。模型在推理时临时学会新任务，关掉就忘了。"
      },
      {
        word: "Task Generalization",
        meaning: "任务泛化",
        breakdown: { root: "Zero-shot" },
        explanation: "模型将已学知识迁移到新任务的能力。如训练时学会英译中，Zero-shot时也能做法译中。背后是对'翻译'这个抽象概念的理解，而非死记硬背特定语言对。"
      }
    ],
    quiz: {
      question: "Zero-shot能力的来源是什么？",
      options: [
        "大量标注数据",
        "预训练学到的广泛知识",
        "专门的任务训练",
        "更多的模型参数"
      ],
      correctAnswer: 1
    }
  },
  {
    id: 19,
    root: "Alignment",
    origin: "Safety",
    meaning: "对齐",
    description: "Alignment是让AI系统的行为与人类价值观和意图对齐，确保AI做人类想让它做的事。核心挑战：AI目标函数（如最大化奖励）可能与人类真实意图不一致。如让AI'让人类开心'，它可能给人类注射多巴胺而非真正帮助。对齐技术包括RLHF（从反馈学习偏好）、Constitutional AI（遵守规则）、Red Teaming（对抗测试）。对齐是AI安全的核心问题，随着AI能力增强越来越重要。",
    examples: [
      {
        word: "Value Loading",
        meaning: "价值观加载",
        breakdown: { root: "Alignment" },
        explanation: "如何把复杂的人类价值观编码到AI系统中。不能简单地说'做好事'，需要定义什么是'好'，如何权衡冲突价值观（隐私vs安全）。这是哲学和技术的交叉难题。"
      },
      {
        word: "Red Teaming",
        meaning: "红队测试",
        breakdown: { root: "Alignment" },
        explanation: "雇佣专门的人员尝试破解AI的安全机制，找出漏洞。如诱导AI生成有害内容、越狱（Jailbreak）、逃避审查。通过红队测试发现和修复问题。"
      },
      {
        word: "Scalable Oversight",
        meaning: "可扩展监督",
        breakdown: { root: "Alignment" },
        explanation: "如何监督比人类更聪明的AI？人类无法直接判断超智能AI的所有行为。解决方案：让AI辅助人类监督AI（如AI帮人类理解另一个AI的推理过程），形成可扩展的监督链条。"
      }
    ],
    quiz: {
      question: "AI对齐要解决的核心问题是什么？",
      options: [
        "提高模型准确率",
        "让AI行为符合人类价值观和意图",
        "减少计算成本",
        "加快推理速度"
      ],
      correctAnswer: 1
    }
  },
  {
    id: 20,
    root: "Scaling Law",
    origin: "Theory",
    meaning: "缩放定律",
    description: "Scaling Law描述模型性能如何随着规模（参数量、数据量、计算量）增长而提升。OpenAI在2020年发现：模型性能与三要素呈幂律关系，且预测性很强。关键发现：模型越大，性能持续提升，没有明显饱和迹象；数据和计算量应与模型大小匹配增长；性能提升相当平滑，可以预测。Scaling Law支撑了'大力出奇迹'的理念，推动GPT-3、GPT-4、Gemini等超大模型的诞生。但边际收益递减，成本指数级上升。",
    examples: [
      {
        word: "Chinchilla Law",
        meaning: "Chinchilla定律",
        breakdown: { root: "Scaling Law" },
        explanation: "DeepMind发现的优化Scaling Law：对于计算预算C，最优配比是模型参数N和训练Token数D大致相等。如70B模型应训练1.4T tokens。这比之前重参数轻数据的做法更优。"
      },
      {
        word: "Bitter Lesson",
        meaning: "痛苦教训",
        breakdown: { root: "Scaling Law" },
        explanation: "Rich Sutton提出的AI发展规律：依赖算力提升的通用方法（搜索、学习）长期来看总是胜过人类知识和特征工程。Scaling Law是这个教训的最新体现。"
      },
      {
        word: "Compute-Optimal",
        meaning: "计算最优",
        breakdown: { root: "Scaling Law" },
        explanation: "在给定计算预算下，如何分配参数量和训练量以获得最佳性能。不是一味追求更大模型，而是找到最优平衡点。Chinchilla就是遵循这个原则训练的。"
      }
    ],
    quiz: {
      question: "Scaling Law告诉我们什么？",
      options: [
        "模型越大训练越慢",
        "性能随模型规模呈可预测的幂律增长",
        "小模型性能更好",
        "数据量不重要"
      ],
      correctAnswer: 1
    }
  }
];
