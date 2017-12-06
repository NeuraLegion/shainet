# SHAInet

SHAInet - stands for Super Human Artifical Intelegance 
is a neural network in pure Crystal

## Installation

Add this to your application's `shard.yml`:

```yaml
dependencies:
  shainet:
    github: NeuraLegion/shainet
```

## Usage

```crystal
require "shainet"
```

TODO: Write usage instructions here

## Development

- [ ] Add multi-channel filtration, cut the string into multiple "segments" and let each channel process different ones.  
- [ ] Add back-prop\r-prop.  
- [ ] Add needed activation functions.  
- https://en.wikipedia.org/wiki/Recurrent_neural_network  
- [ ] Build diverse and relevant tests to check Neural Net capabilities.  
- [ ] Bind and use CUDA  
- [ ] Research compression-on-the-fly as a mean to minimize the size of the payload sample.  


## Contributing

1. Fork it ( https://github.com/NeuraLegion/shainet/fork )
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create a new Pull Request

## Contributors

- [bararchy](https://github.com/bararchy) - creator, maintainer
