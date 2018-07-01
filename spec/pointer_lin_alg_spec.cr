require "./spec_helper"

describe SHAInet do
  puts "############################################################"
  it "check matrix dot product" do
    puts "\n"

    m1_00 = 1.0
    m1_01 = 2.0
    m1_02 = 3.0
    m1_10 = 4.0
    m1_11 = 5.0
    m1_12 = 6.0

    m2_00 = 1.0
    m2_01 = 2.0
    m2_10 = 3.0
    m2_11 = 4.0
    m2_20 = 5.0
    m2_21 = 6.0

    m_out_00 = 0.0
    m_out_01 = 0.0
    m_out_10 = 0.0
    m_out_11 = 0.0

    m_test = 77.0

    # m1 = [[m1_00, m1_01, m1_02],
    #       [m1_10, m1_11, m1_12]]

    # m2 = [[m2_00, m2_01],
    #       [m2_10, m2_11],
    #       [m2_20, m2_21]]

    # m_out = [[m_out_00, m_out_01],
    #          [m_out_10, m_out_11]]

    m1 = SHAInet::PtrMatrix.new(3, 2)
    m2 = SHAInet::PtrMatrix.new(2, 3)
    m_out = SHAInet::PtrMatrix.new(2, 2)

    v1 = SHAInet::PtrMatrix.new(3, 1)
    v2 = SHAInet::PtrMatrix.new(1, 3)
    v_out = SHAInet::PtrMatrix.new(1, 1)

    puts "---"
    puts "empty m1:"
    m1.show

    puts "---"
    puts "empty v1:"
    v1.show
    puts "---"
    puts "empty v2:"
    v2.show

    # Update matrix data
    m1.data[0][0] = pointerof(m1_00)
    m1.data[0][1] = pointerof(m1_01)
    m1.data[0][2] = pointerof(m1_02)
    m1.data[1][0] = pointerof(m1_10)
    m1.data[1][1] = pointerof(m1_11)
    m1.data[1][2] = pointerof(m1_12)

    m2.data[0][0] = pointerof(m2_00)
    m2.data[0][1] = pointerof(m2_01)
    m2.data[1][0] = pointerof(m2_10)
    m2.data[1][1] = pointerof(m2_11)
    m2.data[2][0] = pointerof(m2_20)
    m2.data[2][1] = pointerof(m2_21)

    m_out.data[0][0] = pointerof(m_out_00)
    m_out.data[0][1] = pointerof(m_out_01)
    m_out.data[1][0] = pointerof(m_out_10)
    m_out.data[1][1] = pointerof(m_out_11)

    # Update vector data
    v1.data[0][0] = pointerof(m1_00)
    v1.data[0][1] = pointerof(m1_01)
    v1.data[0][2] = pointerof(m1_02)
    puts "xxx here xxx"

    v2.data[0][0] = pointerof(m2_00)
    v2.data[1][0] = pointerof(m2_10)
    v2.data[2][0] = pointerof(m2_20)

    v_out.data[0][0] = pointerof(m_out_00)

    puts "--- before do product ---"
    puts "m1:"
    m1.show
    puts "---"
    puts "m2:"
    m2.show
    puts "---"
    puts "m_out:"
    m_out.show

    m1.static_dot(m2, m_out)

    puts "--- after do product ---"
    puts "m1:"
    m1.show
    puts "---"
    puts "m2:"
    m2.show
    puts "---"
    puts "m_out:"
    m_out.show

    puts "--- before do product ---"
    puts "v1:"
    v1.show
    puts "---"
    puts "v2:"
    v2.show
    puts "---"
    puts "v_out:"
    v_out.show

    v1.static_dot(v2, v_out)

    puts "--- after do product ---"
    puts "v1:"
    v1.show
    puts "---"
    puts "v2:"
    v2.show
    puts "---"
    puts "v_out:"
    v_out.show
  end

  puts "############################################################"

  # it "check transpose" do
  #   puts "\n"

  #   m1_00 = 1.0
  #   m1_01 = 2.0
  #   m1_02 = 3.0
  #   m1_10 = 4.0
  #   m1_11 = 5.0
  #   m1_12 = 6.0

  #   m2_00 = 1.0
  #   m2_01 = 2.0
  #   m2_10 = 3.0
  #   m2_11 = 4.0
  #   m2_20 = 5.0
  #   m2_21 = 6.0

  #   m_out_00 = 0.0
  #   m_out_01 = 0.0
  #   m_out_10 = 0.0
  #   m_out_11 = 0.0

  #   m_test = 77.0

  #   # m1 = [[m1_00, m1_01, m1_02],
  #   #       [m1_10, m1_11, m1_12]]

  #   # m2 = [[m2_00, m2_01],
  #   #       [m2_10, m2_11],
  #   #       [m2_20, m2_21]]

  #   # m_out = [[m_out_00, m_out_01],
  #   #          [m_out_10, m_out_11]]

  #   m1 = SHAInet::PtrMatrix.new(3, 2)
  #   m2 = SHAInet::PtrMatrix.new(2, 3)
  #   m_out = SHAInet::PtrMatrix.new(2, 2)

  #   puts "---"
  #   puts "empty m1:"
  #   m1.show

  #   m1.data[0][0] = pointerof(m1_00)
  #   m1.data[0][1] = pointerof(m1_01)
  #   m1.data[0][2] = pointerof(m1_02)
  #   m1.data[1][0] = pointerof(m1_10)
  #   m1.data[1][1] = pointerof(m1_11)
  #   m1.data[1][2] = pointerof(m1_12)
  #   puts "---"
  #   puts "updated m1:"
  #   m1.show

  #   m2.data[0][0] = pointerof(m2_00)
  #   m2.data[0][1] = pointerof(m2_01)
  #   m2.data[1][0] = pointerof(m2_10)
  #   m2.data[1][1] = pointerof(m2_11)
  #   m2.data[2][0] = pointerof(m2_20)
  #   m2.data[2][1] = pointerof(m2_21)

  #   m_out.data[0][0] = pointerof(m_out_00)
  #   m_out.data[0][1] = pointerof(m_out_01)
  #   m_out.data[1][0] = pointerof(m_out_10)
  #   m_out.data[1][1] = pointerof(m_out_11)

  #   puts "--- before do product ---"
  #   puts "m1:"
  #   m1.show
  #   puts "---"
  #   puts "m2:"
  #   m2.show
  #   puts "---"
  #   puts "m_out:"
  #   m_out.show

  #   m1.static_dot(m2, m_out)

  #   puts "--- after do product ---"
  #   puts "m1:"
  #   m1.show
  #   puts "---"
  #   puts "m2:"
  #   m2.show
  #   puts "---"
  #   puts "m_out:"
  #   m_out.show
  # end
end
