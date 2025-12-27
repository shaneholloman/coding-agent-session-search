use coding_agent_search::default_data_dir;
use coding_agent_search::search::query::{SearchClient, SearchFilters};
use coding_agent_search::search::tantivy::index_dir;
use coding_agent_search::search::vector_index::{Quantization, VectorEntry, VectorIndex};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn bench_empty_search(c: &mut Criterion) {
    let data_dir = default_data_dir();
    let index_path = index_dir(&data_dir).unwrap();
    let client = SearchClient::open(&index_path, None).unwrap();
    if let Some(client) = client {
        c.bench_function("search_empty_query", |b| {
            b.iter(|| {
                client
                    .search("", SearchFilters::default(), 10, 0)
                    .unwrap_or_default()
            })
        });
    }
}

fn bench_vector_index_search(c: &mut Criterion) {
    let dimension = 384;
    let count = 50_000;
    let entries = build_entries(count, dimension);
    let index = VectorIndex::build(
        "bench-embedder",
        "rev",
        dimension,
        Quantization::F16,
        entries,
    )
    .unwrap();
    let query = build_query(dimension);

    c.bench_function("vector_index_search_50k", |b| {
        b.iter(|| {
            let results = index
                .search_top_k(black_box(&query), 25, None)
                .unwrap_or_default();
            black_box(results);
        });
    });
}

fn build_entries(count: usize, dimension: usize) -> Vec<VectorEntry> {
    let mut entries = Vec::with_capacity(count);
    for idx in 0..count {
        let mut vector = Vec::with_capacity(dimension);
        for d in 0..dimension {
            let value = ((idx + d * 31) % 997) as f32 / 997.0;
            vector.push(value);
        }
        entries.push(VectorEntry {
            message_id: idx as u64,
            created_at_ms: idx as i64,
            agent_id: (idx % 8) as u32,
            workspace_id: 1,
            source_id: 1,
            role: 1,
            chunk_idx: 0,
            content_hash: [0u8; 32],
            vector,
        });
    }
    entries
}

fn build_query(dimension: usize) -> Vec<f32> {
    let mut query = Vec::with_capacity(dimension);
    for d in 0..dimension {
        query.push((d % 17) as f32 / 17.0);
    }
    query
}

criterion_group!(benches, bench_empty_search, bench_vector_index_search);
criterion_main!(benches);
