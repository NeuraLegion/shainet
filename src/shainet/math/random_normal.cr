require "csv"

module SHAInet
  module RandomNormal
    extend self

    # Normal probability density function calculation
    def pdf(x : Float64, mu : Float64, sigma : Float64)
      max_y = Float64.new((1 / Math.sqrt(2 * Math::PI * sigma**2)))
      exp = Float64.new(Math::E**(-1 * (x - mu)**2 / (2 * sigma**2)))
      return max_y*exp
    end

    # Sampling n points from a normal distribution with mu & sigma,
    # using the Metropolis-Hastings algorithm
    def metropolis(n : Int32 = 1, mu : Float64 = 0.0, sigma : Float64 = 1.0)
      points = Array(Float64).new
      r = mu
      p = pdf(x: r, mu: mu, sigma: sigma)

      n.times do
        rn = r.clone + rand(-1.0..1.0)
        pn = pdf(x: rn, mu: mu, sigma: sigma)

        if pn >= p
          p = pn.clone
          r = rn.clone
        else
          u = rand(1.0)
          if u < (pn / p)
            p = pn.clone
            r = rn.clone
          end
        end
        points << r
      end

      return points
    end

    # alias_method :sample, :metropolis
    def sample(n : Int32 = 1, mu : Float64 = 0.0, sigma : Float64 = 1.0)
      raise "Parameter error, sampling must be of n >= 1" if n < 1
      return metropolis(n: n, mu: mu, sigma: sigma)
    end
  end
end

data = SHAInet::RandomNormal.sample(10000, 0.0, 3.0)
csv_file = CSV.build do |csv|
  data.each do |value|
    csv.row value
  end
end

File.write("csv_file.csv", csv_file)
