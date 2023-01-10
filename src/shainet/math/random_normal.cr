module SHAInet
  module RandomNormal
    extend self

    # Normal probability density function calculation
    def pdf(x : Float64, mu : Float64, sigma : Float64)
      max_y = Float64.new((1 / Math.sqrt(2 * Math::PI * sigma**2)))
      exp = Float64.new(Math::E**(-1 * (x - mu)**2 / (2 * sigma**2)))
      max_y * exp
    end

    # Sampling n points from a normal distribution with mu & sigma,
    # using the Metropolis-Hastings algorithm
    def metropolis(n : Int32 = 1, mu : Float64 = 0.0, sigma : Float64 = 1.0)
      raise "Parameter error, sampling must be of n >= 1" if n < 1
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

      points
    end

    def sample(n : Int32 = 1, mu : Float64 = 0.0, sigma : Float64 = 1.0)
      metropolis(n: n, mu: mu, sigma: sigma)
    end
  end
end
