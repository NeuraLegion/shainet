require "./spec_helper"

describe "LLM integration" do
  it "predicts the next token" do
    Random::DEFAULT.new_seed(42_u64, 54_u64)
    corpus = ("the quick brown fox jumps over the lazy dog " * 5).strip
    tokenizer = SHAInet::BPETokenizer.new
    tokenizer.train(corpus, 50)

    quick_id = tokenizer.encode("quick").first
    brown_id = tokenizer.encode("brown").first

    training = [] of Array(Array(Array(Float64)) | Array(Float64))
    50.times do
      seq = [[quick_id.to_f64]]
      target = Array(Float64).new(tokenizer.vocab.size, 0.0)
      target[brown_id] = 1.0
      training << [seq, target]
    end

    net = SHAInet::Network.new
    net.add_layer(:input, 1, :memory, SHAInet.none)
    net.add_layer(:embedding, 8, vocab_size: tokenizer.vocab.size)
    net.add_layer(:lstm, 16)
    net.add_layer(:output, tokenizer.vocab.size, :memory, SHAInet.sigmoid)
    net.fully_connect
    net.learning_rate = 0.01
    net.clip_threshold = 1.0

    net.train(training,
      training_type: :sgdm,
      cost_function: :c_ent,
      epochs: 200,
      mini_batch_size: 1,
      log_each: 200)

    out = net.run([[quick_id]]).last
    pred_id = out.each_with_index.max_by { |v, i| v }[1]
    tokenizer.decode([pred_id]).should eq("brown")
  end
end
