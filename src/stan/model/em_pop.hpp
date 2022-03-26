#ifndef STAN_MODEL_MODEL_EM_POP_HPP
#define STAN_MODEL_MODEL_EM_POP_HPP

#include <stan/io/var_context.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/fun/welford_var_estimator.hpp>
#include <stan/math/prim/fun/welford_covar_estimator.hpp>
#include <boost/random/additive_combine.hpp>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace stan {
namespace model {

  /** 
   * stochastic approximation version of Welford estimator
   * 
   */
class welford_sa_estimator : public stan::math::welford_covar_estimator {
 protected:
  int iter;                     /**< SAEM iteration counter */
  int k1_;                       /**< first SAEM interval cutoff */
  Eigen::VectorXd curr_m_;      /**< weighted/accumulated m */
  Eigen::MatrixXd curr_m2_;     /**< weighted/accumulated m2 */
 public:
  explicit welford_sa_estimator(int n)
      : welford_covar_estimator(n), iter(0),
        k1_(50/n),
        curr_m_(m_), curr_m2_(m2_) {
    restart();
  }

  welford_sa_estimator(int n, int k1)
      : welford_covar_estimator(n), iter(0),
        k1_(k1),
        curr_m_(m_), curr_m2_(m2_) {
    restart();
  }

  int k1() { return k1_; }

  void incr_iter() {
    iter++;
    std::cout << "[torsten debug print] EM iter: " << iter << "\n";
  }

  double gamma(int i) {
    return (i > k1_ ? 1.0/(i - k1_) : 1.0);
  }

  void restart_sa() {
    iter = 0;
    curr_m_.setZero();
    curr_m2_.setZero();
    restart();
  }

  void weighted_sample_estimate(Eigen::VectorXd& mu,
                           Eigen::MatrixXd& covar) {
    sample_mean(mu);
    sample_covariance(covar);
    covar *= (num_samples_-1);
    covar += num_samples_ * mu * mu.transpose() + // sum(x)*mu
      num_samples_ * mu * mu.transpose() -        // mu*sum(x)
      num_samples_ * mu * mu.transpose();         // n * mu * mu
    // add weight to sum(x * x^t)
    covar *= gamma(iter);
    covar += curr_m2_;
    
    // update sufficient statistics
    curr_m2_ = (1.0 - gamma(iter + 1)) * covar;
    
    // add weight to sum(x)
    mu *= num_samples_ * gamma(iter);
    mu += curr_m_;

    // update sufficient statistics
    curr_m_ = (1.0 - gamma(iter + 1)) * mu;

    // from sufficent stat X*X^t to covariance matrix
    covar += -mu * mu.transpose()/num_samples_;
    // covar += -num_samples_ * m_ * mu.transpose() -
    //   num_samples_ * mu * m_.transpose() + num_samples_ * mu * mu.transpose();

    // final sample mean & covar
    mu /= num_samples_;
    covar /= (num_samples_-1);

    assert(covar(0, 0) > 0);
    // assert(covar(1, 1) > 0);
  }
};

class em_pop {
 public:
  Eigen::VectorXd* p_mu;
  Eigen::MatrixXd* p_sigma;
  size_t n_subj;
  size_t n_mu;
  size_t n_cov;
  welford_sa_estimator* covar_est;

  em_pop() : p_mu(nullptr), p_sigma(nullptr), n_subj(0),
             covar_est(nullptr)
  {}

  Eigen::VectorXd& em_population_mean() {
    return *p_mu;
  }

  Eigen::MatrixXd & em_population_covar() {
    return *p_sigma;
  }

  /** 
   * use constrained individual param sample to update population
   * sufficient statistics.
   * 
   */
  void accumulate_ind_param(std::vector<Eigen::Matrix<double, -1, 1>> const& p) {
    Eigen::VectorXd& ref_mu = *p_mu;
    Eigen::MatrixXd& ref_sigma = *p_sigma;

    n_subj = p.size();
    for (auto i = 0; i < n_subj; ++i) {
      covar_est -> add_sample(p[i]);
    }
  }

  void em_step() {
    Eigen::VectorXd& ref_mu = *p_mu;
    Eigen::MatrixXd& ref_sigma = *p_sigma;

    covar_est -> weighted_sample_estimate(ref_mu, ref_sigma);
    covar_est -> incr_iter();
    covar_est -> restart();

    std::cout << "[torsten debug print] EM mean: " << em_population_mean().transpose() << "\n";
    std::cout << "[torsten debug print] EM covar: " << em_population_covar() << "\n";

    // debug
    // ref_sigma << 
    // 0.6259198, -0.2525684,  0.1700814,
    // -0.2525684,  2.0582260, -0.7210698,
    // 0.1700814, -0.7210698,  0.3230868; 
  }
};
}
}

#endif

