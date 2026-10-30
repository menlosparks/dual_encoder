from collections.abc import Sequence
import importlib
import os
from typing import Any


import kagglehub
from absl import app
from absl import flags
import numpy as np
import sentence_transformers
import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAME = 'Qwen/Qwen3-Embedding-0.6B'
QUESTION_SPARSE_DIM = 512
HIDDEN_DIM = 256
EMBEDDING_DIM = 64
## The final shared projection layer    
shared_projection = nn.Linear(HIDDEN_DIM, EMBEDDING_DIM, bias=False)



# Answer tower
## Sentence embedder - the first layer of the answer tower
sentence_embedder = sentence_transformers.SentenceTransformer(MODEL_NAME)

## Freeze the weights of the sentence embedder since it is already pre-trained
## Changing these will cause 'catastrophic forgetting' of language
for param in sentence_embedder.parameters():
      param.requires_grad = False

## The input to the next layer is the output dimension of the sentence embedder
input_embed_dim = sentence_embedder.get_embedding_dimension()

## Layer that projects the output of the sentence embedder to the hidden vector by first passing it through a non-linearity
answer_encoder = nn.Sequential(
        nn.Linear(input_embed_dim, HIDDEN_DIM),
        nn.GELU(),
        nn.LayerNorm(HIDDEN_DIM),
    )

class AnswerTower(nn.Module):
  def __init__(self):
    super().__init__()
    self.sentence_embedder = sentence_embedder
    for param in self.sentence_embedder.parameters():
      param.requires_grad = False

    # Answer encoder tower (text -> 1024-D -> HIDDEN_DIM).
    self.answer_encoder = nn.Sequential(
        nn.Linear(input_embed_dim, HIDDEN_DIM),
        nn.GELU(),
        nn.LayerNorm(HIDDEN_DIM),

    )

    # Common bias-free linear projection layer shared with SparseQuestionEncoder.
    self.shared_projection = shared_projection


  ## Encodes the answer to the final projection
  def encode_answer(
      self,
      answers: Sequence[str] | torch.Tensor,
  ) -> torch.Tensor:
    """Produces the separate L2-normalized Answer Embedding `(B, proj_dim)`."""
    with torch.no_grad():
      embeddings = self.sentence_embedder.encode(
          list(answers),
          convert_to_tensor=True,
          show_progress_bar=False,
      )
    a_sent_emb = embeddings.clone().detach().float()

    h_answer = self.answer_encoder(a_sent_emb)  # (B, HIDDEN_DIM)
    z_answer = self.shared_projection(h_answer)  # (B, EMBEDDING_DIM)
    return F.normalize(z_answer, p=2, dim=-1)


  ## This is the method that handles the step forward in the training process for the answer tower. 
  def forward(
      self,
      answers: Sequence[str] | torch.Tensor,
  ) -> torch.Tensor:
    return self.encode_answer(answers)



answer_tower = AnswerTower()
for name, param in answer_tower.named_parameters():
    print(f"{name:30} | Shape: {str(param.shape):18} | Requires Grad: {param.requires_grad}")

# Question tower

## The first projection layer that projects the question vector to the intermediate hidden layer dimension
## i.e the first embedding layer for the question
question_projection = nn.Linear(QUESTION_SPARSE_DIM, HIDDEN_DIM)  


class QuestionTower(nn.Module):
  def __init__(self):
    super().__init__()

    # Answer encoder tower (text -> 1024-D -> HIDDEN_DIM).
    self.question_projection = question_projection

    # Common bias-free linear projection layer shared with SparseQuestionEncoder.
    self.shared_projection = shared_projection

  ## Takes the sparse user profile vector as the input and returns the final projection layer embedding.
  def forward(
      self,
      questions: Sequence[Sequence[float]],
  ) -> torch.Tensor:
  
      q_tensor = torch.as_tensor(questions, dtype=torch.float32)
      
      h_question = question_projection(q_tensor)  # (B, HIDDEN_DIM)
      z_question = shared_projection(h_question)  # (B, EMBEDDING_DIM)
      return F.normalize(z_question, p=2, dim=-1)


question_tower=QuestionTower()
for name, param in question_tower.named_parameters():
    print(f"{name:30} | Shape: {str(param.shape):18} | Requires Grad: {param.requires_grad}")


# Similarity module

class SimilarityModule(nn.Module):
  def __init__(self):
    super().__init__()

  ## This module take the output o fthe answer and question tower and returns the simlarity calculation.
  ## The cosine function will be used as the similarity measure.
  
  def forward(
      self,
      question_embedding: torch.Tensor,
      answer_embedding: torch.Tensor,
  ) -> torch.Tensor:
    cosine_sim = F.cosine_similarity(
        question_embedding.unsqueeze(1),
        answer_embedding.unsqueeze(0),
        dim=-1,
    )
    return cosine_sim

similarity_module = SimilarityModule()

for name, param in similarity_module.named_parameters():
    print(f"{name:30} | Shape: {str(param.shape):18} | Requires Grad: {param.requires_grad}")



# Load data

answers_file = kagglehub.dataset_download(                                                                                   
        'abhishind/dual-encoder-test',                                                                                           
        path='answers.txt',                                                                                                      
    )                                                                                                                            
                                                                                                                                 
with open(answers_file, 'r', encoding='utf-8') as f:                                                                         
  answers = [line.strip() for line in f if line.strip()]

print(answers)  

questions_file = kagglehub.dataset_download(                                                                                 
        'abhishind/dual-encoder-test',                                                                                           
        path='questions.txt',                                                                                                    
    )                                                                                                                            
                                                                                                                                 
# 2D float32 array of shape (num_lines, 512), where each row questions[i]                                                    
# is a 1D numpy.ndarray of dtype float32:                                                                                    
questions = np.loadtxt(questions_file, delimiter=',', dtype=np.float32)                                                      

print(questions)          


# Define optimizer and set up train mode

unique_trainable_params = list(
    {
        id(p): p
        for p in (
            list(question_tower.parameters())
            + list(answer_tower.parameters())
            + list(similarity_module.parameters())

        )
        if p.requires_grad
    }.values()
)
optimizer = torch.optim.Adam(unique_trainable_params, lr=3e-3)

question_tower.train()
answer_tower.train()
similarity_module.train()


# Loss function
def cosine_similarity_loss(cosine_sim: torch.Tensor) -> torch.Tensor:


  batch_size = cosine_sim.size(0)
  
  
  ## Get a one-dimension vector of all 1's
  pos_targets = torch.ones(batch_size, device=cosine_sim.device)
  
  ## Calculate the MSE loss of the cosine diagonal against the 1's
  ## because this is what we want the diagonal to have.
  diagonal_loss = F.mse_loss(torch.diagonal(cosine_sim), pos_targets)

  ## Get the identity matrix which has 1's  on the diagonal and 0's off diagonal
  target_sim = torch.eye(batch_size, device=cosine_sim.device)
  
  ## Calculate a matrix level loss so that the diagonal is 1's and off-diagonals is 0's
  matrix_loss = F.mse_loss(cosine_sim, target_sim)
  
  ## Then return the total loss
  ## We are effectively including the loss on the diagonal twice since we want to emphasize that matching pairs
  total_loss = diagonal_loss + matrix_loss

  return total_loss



# Run training *loop*

EPOCHS=10

questions_tensor = torch.from_numpy(questions)

for epoch in range(1, EPOCHS + 1):
  optimizer.zero_grad()
  question_embedding = question_tower(questions_tensor)
  answer_embedding = answer_tower(answers)

  cosine_sim = similarity_module(question_embedding, answer_embedding)


  loss = cosine_similarity_loss(cosine_sim)
  loss.backward()
  optimizer.step()

  print(
          f'  Epoch {epoch:02d} loss: {loss.item():.4f} '
          f'(mean_pos_cos={torch.diagonal(cosine_sim).mean().item():+.4f})'
       )





# Load answers to Vector DB
embedding_dim = answer_tower.shared_projection.out_features

## Create a vector database with the output embedding dimension of our model
answer_vector_db = HnswVectorDatabase(embedding_dim=EMBEDDING_DIM)
print(EMBEDDING_DIM)

## Use only the answer tower and put it in evalaution mode (as opposed to training mode)
answer_tower.eval()

## For each answer, create its embedding and save it
## We will add a index number as part of the metadata

with torch.no_grad():
 for idx, answer_text in enumerate(answers):
   answer_emb = answer_tower.encode_answer([answer_text]).cpu().numpy()
   answer_vector_db.add_embeddings(
       embeddings=answer_emb,
       texts=[answer_text],
       item_ids=[idx],
       item_type='positive_answer',
   )
   answer_vector_db._metadata_by_id[idx]['index'] = idx

print(
      f'\nIndexed {answer_vector_db.size} positive answer embeddings '
      f'(dim={answer_vector_db.embedding_dim}) in HNSW vector database.'
  )




# Get top 3 for a user profile

## Sparse user profile vector of dimension `QUESTION_SPARSE_DIM` (512)
user_profile_vector = questions[0]
top_k=3

user_profile_array =  np.asarray([user_profile_vector], dtype=np.float32)

## Put the question tower in eval mode and get the embedding
question_tower.eval()
with torch.no_grad():
    user_profile_embedding = question_tower(user_profile_array).cpu().numpy()[0]


## Get the matches and print out their text and similarity
top_matches = answer_vector_db.search(user_profile_embedding, top_k=top_k)

for rank, match in enumerate(top_matches, start=1):
  item_id = match['id']
  item_text =  match["text"]
  print(
    f'cos_sim={match["cosine_similarity"]:.4f}): '
    f'id={match["id"]}, text="{match["text"]}'
  )
