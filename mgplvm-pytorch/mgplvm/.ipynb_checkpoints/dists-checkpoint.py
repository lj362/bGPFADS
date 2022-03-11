import torch
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, probs_to_logits, lazy_property, logits_to_probs

class NegativeBinomial(Distribution):
    r"""
    Creates a Negative Binomial distribution, i.e. distribution
    of the number of successful independent and identical Bernoulli trials
    before :attr:`total_count` failures are achieved. The probability
    of failure of each Bernoulli trial is :attr:`probs`.
    Args:
        total_count (float or Tensor): non-negative number of negative Bernoulli
            trials to stop, although the distribution is still valid for real
            valued count
        logits (Tensor): Event log-odds for probabilities of success
    """
    arg_constraints = {'total_count': constraints.greater_than_eq(0),
                       'logits': constraints.real}
    support = constraints.nonnegative_integer

    def __init__(self, total_count, logits, validate_args=None):

        #logit = log(p/(1-p))
        #p = e^l / (e^l + 1) = 1/(1 + e^(-l))
        #-logits = logit(1-p)
        self.total_count, self.logits, = broadcast_all(total_count, logits)
        self.total_count = self.total_count.type_as(self.logits)
        self._param = self.logits
        batch_shape = self._param.size()
        super(NegativeBinomial, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(NegativeBinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        if 'probs' in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if 'logits' in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(NegativeBinomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    @property
    def mean(self):
        """mu = pr/(1-p)"""
        return self.total_count * torch.exp(self.logits)

    @property
    def variance(self):
        """var = mu/(1-p) = mu*(1 + mu/r)"""
        return self.mean / torch.sigmoid(-self.logits)

    @property
    def param_shape(self):
        return self._param.size()

    @lazy_property
    def _gamma(self):
        """rate = (1-p)/p"""
        # Note we avoid validating because self.total_count can be zero.
        return torch.distributions.Gamma(concentration=self.total_count,
                                         rate=torch.exp(-self.logits),
                                         validate_args=False)

    def sample(self, sample_shape=torch.Size()):
        """sample lambda = gamma(conc = r, rate = (1-p)/p)
        sample y = Poisson(lambda)"""
        with torch.no_grad():
            rate = self._gamma.sample(sample_shape=sample_shape)
            return torch.poisson(rate)

    def log_prob(self, value):
        """binom(y+r-1, y) (1-p)^r p^y"""

        # r*log(1-p) + y*log(p)
        log_unnormalized_prob = (self.total_count * F.logsigmoid(-self.logits) +
                                 value * F.logsigmoid(self.logits))
        
        #-(gamma(y+r) - gamma(y+1) - gamma(r)) = log(binom(y+r-1, y)^(-1))
        log_normalization = (-torch.lgamma(self.total_count + value) + torch.lgamma(1. + value) +
                             torch.lgamma(self.total_count))
        
        # binom(y+r-1, y) (1-p)^r p^y"""
        return log_unnormalized_prob - log_normalization


