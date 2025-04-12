use half::bf16;
use serde::{Deserialize, Serialize};

/// Segment 由一段文本和其可选的 embedding 向量组成，是 Full-Text Search 和 Vector Search 的基本单元。
/// 根据目前主流 Embedding 模型参数，文本长度不建议超过 512 个 token。
/// 一般情况下模型中 token 和字数的换算比例大致如下：1 个英文字符 ≈ 0.3 个 token；1 个中文字符 ≈ 0.6 个 token。
/// 向量维度一般在 512 以上，维度越高检索精度越高，但消耗的内存和计算资源也越多。
/// 通过一个 Document 应该切分成一组 Segments 进行存储和检索。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Segment {
    #[serde(rename = "i")]
    pub id: u64,

    #[serde(rename = "t")]
    pub text: String,

    #[serde(rename = "v")]
    pub vec: Option<Vec<bf16>>,
}

impl Segment {
    /// 创建一个新的segment
    pub fn new(text: String) -> Self {
        Self {
            id: 0,
            text,
            vec: None,
        }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    /// 获取segment的文本内容
    pub fn text(&self) -> &str {
        &self.text
    }

    /// 获取segment的向量表示
    pub fn vec(&self) -> Option<&Vec<bf16>> {
        self.vec.as_ref()
    }

    pub fn set_id(&mut self, id: u64) -> &Self {
        self.id = id;
        self
    }

    /// 设置segment的向量表示
    pub fn set_vec(&mut self, vec: Vec<bf16>) -> &Self {
        self.vec = Some(vec);
        self
    }
}
