module SHAInet
  module Autograd
    class Tensor
      property data : Float64
      property grad : Float64
      property parents : Array(Tensor)
      property backward_fn : Proc(Float64, Nil)?

      def initialize(@data : Float64, @parents : Array(Tensor) = [] of Tensor, @backward_fn : Proc(Float64, Nil)? = nil)
        @grad = 0.0
      end

      def +(other : Tensor)
        Tensor.new(@data + other.data, [self, other], ->(g : Float64) do
          self.grad += g
          other.grad += g
        end)
      end

      def +(other : Number)
        self + Tensor.new(other.to_f)
      end

      def -(other : Tensor)
        Tensor.new(@data - other.data, [self, other], ->(g : Float64) do
          self.grad += g
          other.grad -= g
        end)
      end

      def -(other : Number)
        self - Tensor.new(other.to_f)
      end

      def *(other : Tensor)
        Tensor.new(@data * other.data, [self, other], ->(g : Float64) do
          self.grad += g * other.data
          other.grad += g * self.data
        end)
      end

      def *(other : Number)
        self * Tensor.new(other.to_f)
      end

      def /(other : Tensor)
        Tensor.new(@data / other.data, [self, other], ->(g : Float64) do
          self.grad += g / other.data
          other.grad -= g * @data / (other.data * other.data)
        end)
      end

      def /(other : Number)
        self / Tensor.new(other.to_f)
      end

      def matmul(other : Tensor)
        Tensor.new(@data * other.data, [self, other], ->(g : Float64) do
          self.grad += g * other.data
          other.grad += g * self.data
        end)
      end

      def backward(initial_grad : Float64 = 1.0)
        build_topology(self, Set(Tensor).new).reverse_each do |t|
          if t == self
            t.grad += initial_grad
          end
          t.backward_fn.try &.call(t.grad)
        end
      end

      private def build_topology(t : Tensor, visited : Set(Tensor), topo = [] of Tensor)
        unless visited.includes?(t)
          visited.add(t)
          t.parents.each { |p| build_topology(p, visited, topo) }
          topo << t
        end
        topo
      end
    end
  end
end
