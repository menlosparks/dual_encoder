"""HNSW cosine similarity vector database wrapper backed by `hnswlib`."""
!pip install hnswlib
from collections.abc import Sequence
import json
import os
from typing import Any

import hnswlib
import numpy as np

EMBEDDING_DIM_DEFAULT = 128


class HnswVectorDatabase:
  """Vector database backed by `hnswlib` using cosine similarity."""

  def __init__(
      self,
      embedding_dim: int = EMBEDDING_DIM_DEFAULT,
      max_elements: int = 10000,
      ef_construction: int = 200,
      m_parameter: int = 16,
      initialize_empty: bool = True,
  ):
    """Initializes the instance.

    Args:
      embedding_dim: Dimensionality of stored vectors (128).
      max_elements: Maximum capacity of the HNSW index.
      ef_construction: HNSW construction accuracy/speed parameter.
      m_parameter: HNSW bidirectional links per node.
      initialize_empty: Whether to initialize a fresh empty HNSW index (`False`
        when loading a serialized index from disk).
    """
    self.embedding_dim = embedding_dim
    self.max_elements = max_elements
    self.ef_construction = ef_construction
    self.m_parameter = m_parameter
    self.index = hnswlib.Index(space='cosine', dim=embedding_dim)
    if initialize_empty:
      self.index.init_index(
          max_elements=max_elements,
          ef_construction=ef_construction,
          M=m_parameter,
      )
      self.index.set_ef(50)
    self._documents_by_id: dict[int, str] = {}
    self._metadata_by_id: dict[int, dict[str, Any]] = {}

  def add_embeddings(
      self,
      embeddings: np.ndarray,
      texts: Sequence[str],
      item_ids: Sequence[int] | None = None,
      item_type: str = 'document',
  ) -> list[int]:
    """Stores 128-D embeddings and their associated text strings in HNSW.

    Args:
      embeddings: 2D float32 NumPy array of shape `(num_items, embedding_dim)`.
      texts: Sequence of text strings corresponding to each embedding row.
      item_ids: Optional explicit integer IDs. If `None`, sequential IDs are
        assigned starting from the current database size.
      item_type: Metadata label describing the entry type (e.g., `'document'`).

    Returns:
      List of integer IDs assigned to the stored embeddings.

    Raises:
      ValueError: If `embeddings` shape does not match `(len(texts), dim)`.
    """
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != self.embedding_dim:
      raise ValueError(
          f'Expected embeddings shape (N, {self.embedding_dim}), '
          f'got {vectors.shape}.'
      )
    if vectors.shape[0] != len(texts):
      raise ValueError(
          f'Number of embeddings ({vectors.shape[0]}) does not match '
          f'number of texts ({len(texts)}).'
      )

    if item_ids is None:
      start_id = len(self._documents_by_id)
      assigned_ids = list(range(start_id, start_id + len(texts)))
    else:
      assigned_ids = list(item_ids)

    self.index.add_items(
        vectors, np.asarray(assigned_ids, dtype=np.int64)
    )
    for item_id, text_value, vector in zip(
        assigned_ids, texts, vectors, strict=True
    ):
      self._documents_by_id[item_id] = text_value
      self._metadata_by_id[item_id] = {
          'id': item_id,
          'type': item_type,
          'text': text_value,
          'embedding': vector.tolist(),
      }
    return assigned_ids

  def search(
      self,
      query_embedding: np.ndarray,
      top_k: int = 3,
  ) -> list[dict[str, Any]]:
    """Searches the HNSW vector database for the closest embeddings by cosine.

    Args:
      query_embedding: 1D array of shape `(embedding_dim,)` or 2D array of
        shape `(1, embedding_dim)`.
      top_k: Number of nearest neighbors to retrieve.

    Returns:
      List of result dictionaries sorted by descending cosine similarity, each
      containing `'id'`, `'text'`, `'cosine_similarity'`, and
      `'cosine_distance'`.
    """
    query_vec = np.asarray(query_embedding, dtype=np.float32)
    if query_vec.ndim == 1:
      query_vec = np.expand_dims(query_vec, axis=0)
    effective_k = min(top_k, len(self._documents_by_id))
    labels, distances = self.index.knn_query(query_vec, k=effective_k)

    results: list[dict[str, Any]] = []
    for doc_id, cosine_dist in zip(labels[0], distances[0], strict=True):
      resolved_id = int(doc_id)
      # In hnswlib with space='cosine', distance = 1.0 - cosine_similarity.
      cosine_sim = float(1.0 - cosine_dist)
      results.append({
          'id': resolved_id,
          'text': self._documents_by_id[resolved_id],
          'cosine_similarity': cosine_sim,
          'cosine_distance': float(cosine_dist),
      })
    return results

  def get_embedding_by_id(self, item_id: int) -> np.ndarray:
    """Returns the stored 128-D embedding vector for `item_id`."""
    return np.asarray(
        self._metadata_by_id[item_id]['embedding'], dtype=np.float32
    )

  @property
  def size(self) -> int:
    """Returns the number of embeddings stored in the vector database."""
    return len(self._documents_by_id)

  def save(self, directory_path: str, prefix: str = 'documents') -> None:
    """Saves the HNSW index and associated metadata to `directory_path`.

    Args:
      directory_path: Target directory where index files will be written.
      prefix: Filename prefix for the HNSW binary index and JSON metadata.
    """
    os.makedirs(directory_path, exist_ok=True)
    index_path = os.path.join(directory_path, f'{prefix}_hnsw.bin')
    metadata_path = os.path.join(directory_path, f'{prefix}_metadata.json')
    self.index.save_index(index_path)
    payload = {
        'embedding_dim': self.embedding_dim,
        'max_elements': self.max_elements,
        'items': list(self._metadata_by_id.values()),
    }
    with open(metadata_path, 'w', encoding='utf-8') as meta_file:
      json.dump(payload, meta_file, indent=2)

  @classmethod
  def load(
      cls,
      directory_path: str,
      prefix: str = 'documents',
  ) -> 'HnswVectorDatabase':
    """Loads a persisted HNSW index and metadata from `directory_path`.

    Args:
      directory_path: Directory containing the saved HNSW index and metadata.
      prefix: Filename prefix used when saving the database.

    Returns:
      Loaded `HnswVectorDatabase` instance.
    """
    index_path = os.path.join(directory_path, f'{prefix}_hnsw.bin')
    metadata_path = os.path.join(directory_path, f'{prefix}_metadata.json')
    with open(metadata_path, 'r', encoding='utf-8') as meta_file:
      payload = json.load(meta_file)

    vector_db = cls(
        embedding_dim=int(payload['embedding_dim']),
        max_elements=int(payload['max_elements']),
        initialize_empty=False,
    )
    vector_db.index.load_index(
        index_path, max_elements=int(payload['max_elements'])
    )
    vector_db.index.set_ef(50)
    for item in payload['items']:
      item_id = int(item['id'])
      vector_db._documents_by_id[item_id] = item['text']
      vector_db._metadata_by_id[item_id] = item
    return vector_db
