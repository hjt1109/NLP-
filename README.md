# NLP-笔记

第一节 机器学习中几个概念解释：

1.错误率
是指模型预测错误的比例，计算方法是：错误率= 错误的预测数/总的预测数
​
2. 精度 (Accuracy)
精度是指模型预测正确的比例
精度=正确的预测数/总的预测数

3.精确率
精确度衡量的是模型预测为正类时，实际为正类的比例。
精确度=true positive /true positive +false positive

4.召回率 (Recall)
召回率衡量的是所有实际为正类的样本中，模型正确预测为正类的比例
召回率=true positive /true positive +false negative 


5.F1 值 (F1 Score)
F1 值是精确度和召回率的调和平均数，它在需要平衡精度和召回率时特别有用
F1值  = 1/(1/精确度+1/召回率) = 2 × 精确度×召回率/（精确度+召回率）

6.AUC (Area Under Curve)
AUC 是 ROC 曲线下的面积，通常用来衡量分类器的整体表现。AUC 值的范围是 [0, 1]，越接近 1，说明模型的  分类性能   越好。
AUC 是通过 ROC 曲线来计算的，它是一个比较全面的评价指标，尤其在类别不平衡的情况下非常有用。



7. ROC (Receiver Operating Characteristic Curve)
ROC 曲线是一个二维图，展示了不同阈值下，分类器的 召回率（真正率）和 假阳性率（假正率）之间的关系。ROC 曲线通过改变分类决策的阈值来查看模型的性能。

真正率 (True Positive Rate, TPR): 即召回率，表示实际为正类的样本中被正确预测为正类的比例。

假阳性率 (False Positive Rate, FPR): 表示实际为负类的样本中被错误预测为正类的比例。
通过画出 ROC 曲线，可以直观地看到模型在各种阈值下的表现

总结
错误率：错误预测的比例。

精度：正确预测的比例。

精确度：预测为正类时，实际为正类的比例。

召回率：实际为正类时，正确预测为正类的比例。

F1 值：精确度和召回率的调和平均。

AUC：ROC 曲线下的面积，表示模型整体性能。

ROC 曲线：展示不同阈值下真正率和假阳性率的关系。



第二节  词嵌入

在NLP(自然语言处理)领域，文本表示是第一步，也是很重要的一步，通俗来说就是把人类的语言符号转化为机器能够进行计算的数字，因为普通的文本语言机器是看不懂的，必须通过转化来表征对应文本。早期是基于规则的方法进行转化，而现代的方法是基于统计机器学习的方法。

文本表示的分类：1.离散表示 2.分布式表示 3. 神经网络表示

1.离散表示
1.1   One-hot表示/编码：
One-hot简称读热向量编码，也是特征工程中最常用的方法。其步骤如下：

构造文本分词后的字典，每个分词是一个比特值，比特值为0或者1。
每个分词的文本表示为该分词的比特位为1，其余位为0的矩阵表示。
例如：John likes to watch movies. Mary likes too

John also likes to watch football games.

以上两句可以构造一个词典，**{"John": 1, "likes": 2, "to": 3, "watch": 4, "movies": 5, "also": 6, "football": 7, "games": 8, "Mary": 9, "too": 10} **

每个词典索引对应着比特位。那么利用One-hot表示为：

**John: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] **

likes: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] .......等等，以此类推。

One-hot表示文本信息的缺点：

随着语料库的增加，数据特征的维度会越来越大，产生一个维度很高，又很稀疏的矩阵。
这种表示方法的分词顺序和在句子中的顺序是无关的，不能保留词与词之间的关系信息。


1.2 词袋模型
文档的向量表示可以直接将各词的词向量表示加和。例如：

John likes to watch movies. Mary likes too

John also likes to watch football games.

以上两句可以构造一个词典，**{"John": 1, "likes": 2, "to": 3, "watch": 4, "movies": 5, "also": 6, "football": 7, "games": 8, "Mary": 9, "too": 10} **

那么第一句的向量表示为：[1,2,1,1,1,0,0,0,1,1]，其中的2表示likes在该句中出现了2次，依次类推。

词袋模型同样有一下缺点：

词向量化后，词与词之间是有大小关系的，不一定词出现的越多，权重越大。
词与词之间是没有顺序关系的。

1.3 TF-IDF
TF-IDF 
TF-IDF（term frequency–inverse document frequency）是一种用于信息检索与数据挖掘的常用加权技术。TF意思是词频(Term Frequency)，IDF意思是逆文本频率指数(Inverse Document Frequency)。
字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。一个词语在一篇文章中出现次数越多, 同时在所有文档中出现次数越少, 越能够代表该文章。



分母之所以加1，是为了避免分母为0。

那么，，从这个公式可以看出，当w在文档中出现的次数增大时，而TF-IDF的值是减小的，所以也就体现了以上所说的了。

**缺点：**还是没有把词与词之间的关系顺序表达出来。

1.4 n-gram 模型

n-gram模型为了保持词的顺序，做了一个滑窗的操作，这里的n表示的就是滑窗的大小，例如2-gram模型，也就是把2个词当做一组来处理，然后向后移动一个词的长度，再次组成另一组词，把这些生成一个字典，按照词袋模型的方式进行编码得到结果。改模型考虑了词的顺序。

例如：

John likes to watch movies. Mary likes too

John also likes to watch football games.

以上两句可以构造一个词典，{"John likes”: 1, "likes to”: 2, "to watch”: 3, "watch movies”: 4, "Mary likes”: 5, "likes too”: 6, "John also”: 7, "also likes”: 8, “watch football”: 9, "football games": 10}

那么第一句的向量表示为：[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]，其中第一个1表示John likes在该句中出现了1次，依次类推。

**缺点：**随着n的大小增加，词表会成指数型膨胀，会越来越大。

1.5 离散表示存在的问题

由于存在以下的问题，对于一般的NLP问题，是可以使用离散表示文本信息来解决问题的，但对于要求精度较高的场景就不适合了。

无法衡量词向量之间的关系。
词表的维度随着语料库的增长而膨胀。
n-gram词序列随语料库增长呈指数型膨胀，更加快。
离散数据来表示文本会带来数据稀疏问题，导致丢失了信息，与我们生活中理解的信息是不一样的。

2.分布式表示
用一个词附近的其它词来表示该词，这是现代统计自然语言处理中最有创见的想法之一。**当初科学家发明这种方法是基于人的语言表达，认为一个词是由这个词的周边词汇一起来构成精确的语义信息。就好比，物以类聚人以群分，如果你想了解一个人，可以通过他周围的人进行了解，因为周围人都有一些共同点才能聚集起来。

2.1 共现矩阵

共现矩阵顾名思义就是共同出现的意思，词文档的共现矩阵主要用于发现主题(topic)，用于主题模型，如LSA。

局域窗中的word-word共现矩阵可以挖掘语法和语义信息，例如：

I like deep learning.
I like NLP.
I enjoy flying
有以上三句话，设置滑窗为2，可以得到一个词典：{"I like","like deep","deep learning","like NLP","I enjoy","enjoy flying","I like"}。

我们可以得到一个共现矩阵(对称矩阵)：

![PixPin_2025-05-12_19-55-16](https://github.com/user-attachments/assets/6975f781-bbd5-4294-858c-c4031ef75891)

中间的每个格子表示的是行和列组成的词组在词典中共同出现的次数，也就体现了共现的特性。

存在的问题：

向量维数随着词典大小线性增长。
存储整个词典的空间消耗非常大。
一些模型如文本分类模型会面临稀疏性问题。
模型会欠稳定，每新增一份语料进来，稳定性就会变化。

3. 神经网络表示

3.1 NNLM 
NNLM (Neural Network Language model)，神经网络语言模型是03年提出来的，通过训练得到中间产物--词向量矩阵，这就是我们要得到的文本表示向量矩阵。

NNLM说的是定义一个前向窗口大小，其实和上面提到的窗口是一个意思。把这个窗口中最后一个词当做y，把之前的词当做输入x，通俗来说就是预测这个窗口中最后一个词出现概率的模型。
![image](https://github.com/user-attachments/assets/dd577f10-fe03-4d7c-a7e9-a65762d32e90)

![image](https://github.com/user-attachments/assets/1b5c00ca-4c52-4e83-a7a3-75182fec3415)

input层是一个前向词的输入，是经过one-hot编码的词向量表示形式，具有V*1的矩阵。

C矩阵是投影矩阵，也就是稠密词向量表示，在神经网络中是w参数矩阵，该矩阵的大小为D*V，正好与input层进行全连接(相乘)得到D*1的矩阵，采用线性映射将one-hot表示投影到稠密D维表示。

![image](https://github.com/user-attachments/assets/86127af1-1999-44e9-8c25-17b62babb09d)

output层(softmax)自然是前向窗中需要预测的词。

通过BP＋SGD得到最优的C投影矩阵，这就是NNLM的中间产物，也是我们所求的文本表示矩阵，通过NNLM将稀疏矩阵投影到稠密向量矩阵中

BP (Backpropagation, 反向传播)：

反向传播是一种用于训练神经网络的算法，它通过计算损失函数的梯度，逐层更新神经网络中各个参数的权重。

具体来说，反向传播的过程从输出层开始，通过计算输出层的误差，然后根据链式法则将误差逐层传递回输入层。在每一层中，误差被用于调整该层的权重，以减少总体的损失。

反向传播算法是深度学习中非常重要的组成部分，尤其是在训练多层神经网络时。

SGD (Stochastic Gradient Descent, 随机梯度下降)：

随机梯度下降是一种优化算法，用于通过梯度下降法最小化损失函数。与传统的梯度下降法（批量梯度下降）不同，SGD每次更新权重时并不是使用整个数据集的梯度，而是使用一个小批量的数据点（或者甚至是单个样本）来估算梯度。

由于SGD基于小批量数据，它的计算效率通常更高，而且能在一些情况下更好地避免陷入局部最小值。

在神经网络的训练中，SGD会使用计算得到的梯度来更新网络的参数，从而不断减少损失函数的值，使网络逐步优化。


3.2 Word2Vec

谷歌2013年提出的Word2Vec是目前最常用的词嵌入模型之一。Word2Vec实际是一种浅层的神经网络模型，它有两种网络结构，**分别是CBOW（Continues Bag of Words）连续词袋和Skip-gram。**Word2Vec和上面的NNLM很类似，但比NNLM简单

CBOW

CBOW获得中间词两边的的上下文，然后用周围的词去预测中间的词，把中间词当做y，把窗口中的其它词当做x输入，x输入是经过one-hot编码过的，然后通过一个隐层进行求和操作，最后通过激活函数softmax，可以计算出每个单词的生成概率，接下来的任务就是训练神经网络的权重，使得语料库中所有单词的整体生成概率最大化，而求得的权重矩阵就是文本表示词向量的结果。


Skip-gram：

Skip-gram是通过当前词来预测窗口中上下文词出现的概率模型，把当前词当做x，把窗口中其它词当做y，依然是通过一个隐层接一个Softmax激活函数来预测其它词的概率。如下图所示：

![image](https://github.com/user-attachments/assets/6e5f56b5-bf2e-43e1-a97e-ae4daae26040)

优化方法：

层次Softmax：至此还没有结束，因为如果单单只是接一个softmax激活函数，计算量还是很大的，有多少词就会有多少维的权重矩阵，所以这里就提出层次Softmax(Hierarchical Softmax)，使用Huffman Tree来编码输出层的词典，相当于平铺到各个叶子节点上，瞬间把维度降低到了树的深度，可以看如下图所示。这课Tree把出现频率高的词放到靠近根节点的叶子节点处，每一次只要做二分类计算，计算路径上所有非叶子节点词向量的贡献即可。
哈夫曼树(Huffman Tree)：给定N个权值作为N个叶子结点，构造一棵二叉树，若该树的带权路径长度达到最小，称这样的二叉树为最优二叉树，也称为哈夫曼树(Huffman Tree)。哈夫曼树是带权路径长度最短的树，权值较大的结点离根较近。


Word2Vec存在的问题

对每个local context window单独训练，没有利用包 含在global co-currence矩阵中的统计信息。
对多义词无法很好的表示和处理，因为使用了唯一的词向量 相当于还是使用了one-hot编码

3.3 sense2vec

Sense2Vec:
目标：通过为单词的每个语义（sense）学习一个独立的向量来处理多义词问题。

方法：Sense2Vec 的基本思想是在 Word2Vec 的基础上扩展，为每个单词的不同语义（sense）学习不同的向量，而不是为每个词学习单一的向量表示。它通过对单词的上下文进行标注，从而区分同一单词的不同意义。

例如，对于 "bank"，Sense2Vec 会生成两个不同的向量表示：一个表示 "银行" 的语义，另一个表示 "河岸" 的语义。

特点：

语义区分：通过对上下文进行细致的分析，Sense2Vec 能够为每个单词的不同意义学习不同的向量。

上下文敏感：它能处理多义词问题，使得相同单词在不同语境中能够有不同的表示，提供更精确的语义信息。

举个例子：
Word2Vec：对于单词 "bank"，无论其在句子中指的是 "金融机构" 还是 "河岸"，Word2Vec 都会给它一个相同的向量表示。

Sense2Vec：对于 "bank"，Sense2Vec 会根据上下文区分它的语义，可能为 "银行" 和 "河岸" 学习两个不同的向量表示。


4. one-hot编码不是一个好的选择，one-hot词向量⽆法准确表达不同词之间的相似度，如我们常常使⽤的余弦相似度。由于任何两个不同词的one-hot向量的余弦相似度都为0，多个不同词之间的相似度难以通过one-hot向量准确地体现出来。

word2vec⼯具的提出正是为了解决上⾯这个问题。它将每个词表⽰成⼀个定⻓的向量，并使得这些向量能较好地表达不同词之间的相似和类⽐关系。

5.word2vec的代码实现
import gensim\n
from gensim.models import Word2Vec
import logging

#设置日志以便观察gensim训练过程
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#示例语料，真实应用中应该用更大的语料库
sentences = [
    ['dog', 'barks'],
    ['cat', 'meows'],
    ['dog', 'chases', 'cat'],
    ['man', 'drives', 'car'],
    ['woman', 'drives', 'car'],
    ['dog', 'runs'],
    ['cat', 'sleeps']
]

#创建Word2Vec模型，使用Skip-gram模型架构
model = Word2Vec(sentences, vector_size=5, window=5, min_count=1, sg=1)

#训练模型
model.save("word2vec.model")
#使用训练好的模型进行相似度查询
similarity = model.wv.most_similar('dog', topn=3)
print("Most similar to 'dog':", similarity)

#查看'cat'的词向量
vector_cat = model.wv['cat']
print("Vector for 'cat':", vector_cat)

#保存词向量到文件
model.wv.save_word2vec_format('word2vec.txt', binary=False)

![image](https://github.com/user-attachments/assets/009500cc-32ee-4b6a-8e5b-91fabfc0de82)

















