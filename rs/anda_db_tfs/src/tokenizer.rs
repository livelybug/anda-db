use rayon::prelude::*;
use std::collections::HashMap;

pub use tantivy_tokenizer_api::*;

/// A chain of tokenizers and filters implemented Tokenizer trait.
pub struct TokenizerChain {
    tokenizer: Box<dyn BoxableTokenizer>,
}

impl TokenizerChain {
    /// Creates a new `TokenizerChainBuilder` with the given tokenizer
    pub fn builder<T: Tokenizer>(tokenizer: T) -> TokenizerChainBuilder<T> {
        TokenizerChainBuilder::new(tokenizer)
    }
}

/// Builder for `TokenizerChain`
pub struct TokenizerChainBuilder<T = Box<dyn BoxableTokenizer>> {
    tokenizer: T,
}

impl<T: Tokenizer> TokenizerChainBuilder<T> {
    /// Creates a new `TokenizerChainBuilder` with the given tokenizer
    pub fn new(tokenizer: T) -> Self {
        TokenizerChainBuilder { tokenizer }
    }

    /// Adds a token filter to the chain
    pub fn filter<F: TokenFilter>(self, token_filter: F) -> TokenizerChainBuilder<F::Tokenizer<T>> {
        TokenizerChainBuilder {
            tokenizer: token_filter.transform(self.tokenizer),
        }
    }

    /// Builds the `TokenizerChain`
    pub fn build(self) -> TokenizerChain {
        TokenizerChain {
            tokenizer: Box::new(self.tokenizer),
        }
    }
}

impl Clone for TokenizerChain {
    fn clone(&self) -> Self {
        TokenizerChain {
            tokenizer: self.tokenizer.box_clone(),
        }
    }
}

/// Implement Tokenizer trait for TokenizerChain
impl Tokenizer for TokenizerChain {
    /// The token stream returned by this Tokenizer.
    type TokenStream<'a> = BoxTokenStream<'a>;

    /// Creates a token stream for a given `str`.
    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        self.tokenizer.box_token_stream(text)
    }
}

/// A boxable `Tokenizer`, with its `TokenStream` type erased.
pub trait BoxableTokenizer: 'static + Send + Sync {
    /// Creates a boxed token stream for a given `str`.
    fn box_token_stream<'a>(&'a mut self, text: &'a str) -> BoxTokenStream<'a>;
    /// Clone this tokenizer.
    fn box_clone(&self) -> Box<dyn BoxableTokenizer>;
}

impl<T: Tokenizer> BoxableTokenizer for T {
    fn box_token_stream<'a>(&'a mut self, text: &'a str) -> BoxTokenStream<'a> {
        BoxTokenStream::new(self.token_stream(text))
    }
    fn box_clone(&self) -> Box<dyn BoxableTokenizer> {
        Box::new(self.clone())
    }
}

/// Creates a new `TokenizerChain` with `SimpleTokenizer` as the default tokenizer and `RemoveLongFilter`, `LowerCaser` and `Stemmer` as the default filters.
#[cfg(any(test, feature = "tantivy"))]
pub fn default_tokenizer() -> TokenizerChain {
    use tantivy::tokenizer::{LowerCaser, RemoveLongFilter, SimpleTokenizer, Stemmer};

    TokenizerChain::builder(SimpleTokenizer::default())
        .filter(RemoveLongFilter::limit(32))
        .filter(LowerCaser)
        .filter(Stemmer::default())
        .build()
}

/// Creates a new `TokenizerChain` with `JiebaTokenizer` as the default tokenizer and `RemoveLongFilter`, `LowerCaser` and `Stemmer` as the default filters.
#[cfg(any(test, feature = "tantivy-jieba"))]
pub fn jieba_tokenizer() -> TokenizerChain {
    use tantivy::tokenizer::{LowerCaser, RemoveLongFilter, Stemmer};
    use tantivy_jieba::JiebaTokenizer;

    TokenizerChain::builder(JiebaTokenizer {})
        .filter(RemoveLongFilter::limit(32))
        .filter(LowerCaser)
        .filter(Stemmer::default())
        .build()
}

/// Tokenizes text and optionally filters tokens
///
/// # Arguments
///
/// * `tokenizer` - Tokenizer to use for processing text
/// * `text` - Text to tokenize
/// * `inclusive` - Optional list of tokens to include (if provided, only tokens in this list will be returned)
///
/// # Returns
///
/// A HashMap of tokens and their counts.
/// The keys are the token strings and the values are their counts.
pub fn collect_tokens<T: Tokenizer>(
    tokenizer: &mut T,
    text: &str,
    inclusive: Option<&HashMap<String, usize>>,
) -> HashMap<String, usize> {
    let mut stream = tokenizer.token_stream(text);
    let mut tokens = HashMap::new();
    while let Some(token) = stream.next() {
        if let Some(inclusive) = inclusive {
            if !inclusive.contains_key(&token.text) {
                continue;
            }
        }

        *tokens.entry(token.text.to_owned()).or_default() += 1;
    }
    tokens
}

const MIN_PARALLEL_SIZE: usize = 1024 * 10; // 10KB

/// Tokenizes text in parallel using multiple threads
///
/// # Arguments
///
/// * `tokenizer` - Tokenizer to use for processing text
/// * `text` - Text to tokenize, will be split by "\n\n"
/// * `inclusive` - Optional list of tokens to include (if provided, only tokens in this list will be returned)
///
/// # Returns
///
/// A HashMap of tokens and their counts.
/// The keys are the token strings and the values are their counts.
pub fn collect_tokens_parallel<T: Tokenizer + Send>(
    tokenizer: &mut T,
    text: &str,
    inclusive: Option<&HashMap<String, usize>>,
) -> HashMap<String, usize> {
    if text.len() < MIN_PARALLEL_SIZE {
        return collect_tokens(tokenizer, text, inclusive);
    }

    let chunks: Vec<&str> = text.split("\n\n").collect();
    let tokens: Vec<HashMap<String, usize>> = chunks
        .par_iter()
        .map(|chunk| {
            let mut local_tokenizer = tokenizer.clone();
            collect_tokens(&mut local_tokenizer, chunk, inclusive)
        })
        .collect();

    tokens
        .into_iter()
        .fold(HashMap::new(), |mut acc, token_map| {
            for (token, count) in token_map {
                *acc.entry(token).or_default() += count;
            }
            acc
        })
}

/// Performs a simple full-text search by finding matching tokens in a document
///
/// # Arguments
///
/// * `tokenizer` - Tokenizer to use for processing text
/// * `query` - Search query text
/// * `doc_text` - Document text to search in
///
/// # Returns
///
/// A HashMap of matching tokens and their counts.
/// The keys are the token strings and the values are their counts.
pub fn flat_full_text_search<T: Tokenizer>(
    tokenizer: &mut T,
    query: &str,
    doc_text: &str,
) -> HashMap<String, usize> {
    let tokens = collect_tokens(tokenizer, query, None);
    collect_tokens_parallel(tokenizer, doc_text, Some(&tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_full_text_search() {
        let mut tokenizer = default_tokenizer();

        // 测试基本匹配
        let matches = flat_full_text_search(
            &mut tokenizer,
            "fox dog",
            "The quick brown fox jumps over the lazy dog",
        );
        assert_eq!(matches.len(), 2);
        assert_eq!(matches.get("fox"), Some(&1));
        assert_eq!(matches.get("dog"), Some(&1));

        // 测试部分匹配
        let matches = flat_full_text_search(
            &mut tokenizer,
            "fox cat",
            "The quick brown fox jumps over the lazy dog",
        );
        assert_eq!(matches.len(), 1);
        assert!(matches.contains_key("fox"));

        // 测试无匹配
        let matches = flat_full_text_search(
            &mut tokenizer,
            "elephant giraffe",
            "The quick brown fox jumps over the lazy dog",
        );
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_collect_tokens() {
        let mut tokenizer = default_tokenizer();

        // 测试基本分词
        let tokens = collect_tokens(&mut tokenizer, "The quick brown fox", None);
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens.get("the"), Some(&1));
        assert_eq!(tokens.get("quick"), Some(&1));
        assert_eq!(tokens.get("brown"), Some(&1));
        assert_eq!(tokens.get("fox"), Some(&1));

        // 测试过滤分词
        let inclusive = HashMap::from([("quick".to_string(), 1), ("fox".to_string(), 1)]);
        let tokens = collect_tokens(
            &mut tokenizer,
            "The quick brown fox, foxes",
            Some(&inclusive),
        );
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens.get("quick"), Some(&1));
        assert_eq!(tokens.get("fox"), Some(&2));
    }

    #[test]
    fn test_collect_tokens_parallel() {
        let mut tokenizer = default_tokenizer();

        // 创建一个大文本，确保触发并行处理
        let large_text =
            "The quick brown fox\n\njumps over\n\nthe lazy dog".repeat(MIN_PARALLEL_SIZE);

        // 测试并行分词
        let tokens = collect_tokens_parallel(&mut tokenizer, &large_text, None);
        assert!(!tokens.is_empty());
        assert!(tokens.contains_key("quick"));
        assert!(tokens.contains_key("fox"));
        assert!(tokens.contains_key("jump"));
        assert!(tokens.contains_key("dog"));

        // 测试小文本（应该使用串行处理）
        let small_text = "The quick brown fox";
        let tokens = collect_tokens_parallel(&mut tokenizer, small_text, None);
        assert_eq!(tokens.len(), 4);
    }
}
