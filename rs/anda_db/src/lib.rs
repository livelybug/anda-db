pub mod collection;
pub mod error;
pub mod schema;
pub mod storage;

// Returns the current unix timestamp in milliseconds.
// #[inline]
// pub fn unix_ms() -> u64 {
//     use std::time::{SystemTime, UNIX_EPOCH};
//
//     let ts = SystemTime::now()
//         .duration_since(UNIX_EPOCH)
//         .expect("system time before Unix epoch");
//     ts.as_millis() as u64
// }
