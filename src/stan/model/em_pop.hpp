#ifndef STAN_MODEL_MODEL_EM_POP_HPP
#define STAN_MODEL_MODEL_EM_POP_HPP

#include <stan/io/var_context.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/fun/welford_var_estimator.hpp>
#include <stan/math/prim/fun/welford_covar_estimator.hpp>
#include <stan/math/rev/functor/gradient.hpp>
#include <stan/math/rev/functor/jacobian.hpp>
#include <boost/random/additive_combine.hpp>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace stan {
namespace model {

  template<typename F_mu, typename T_covariate>
  struct mu_referenced_multi_normal_lpdf {
    F_mu const& f;
    std::vector<T_covariate> const& covariate;
    std::vector<Eigen::VectorXd> const& subj_sample;
    Eigen::MatrixXd const& sigma;

    mu_referenced_multi_normal_lpdf(F_mu const& f0, 
                                    std::vector<T_covariate> const& c,
                                    std::vector<Eigen::VectorXd > const& s,
                                    Eigen::MatrixXd const& sigma0) :
      f(f0), covariate(c), subj_sample(s), sigma(sigma0) 
     {}

    template<typename T>
    T operator()(Eigen::Matrix<T, -1, 1> const& theta) {
      T l = 0;
      size_t n_subj = subj_sample.size();
      for (auto i = 0; i < n_subj; ++i) {
        l += multi_normal_lpdf(subj_sample[i], f(theta, covariate[i]), sigma);
      }
      return l;
    }
  };

  template<typename F_mu, typename T_covariate>
  struct mu_referenced_f {
    F_mu const& f;              /**< mu-ref functor */
    T_covariate const& covariate;

    mu_referenced_f(F_mu const& f0, T_covariate const& c) :
      f(f0), covariate(c)
    {}

    template<typename T>
    Eigen::Matrix<T, -1, 1> operator()(Eigen::Matrix<T, -1, 1> const& theta) {
      return f(theta, covariate);
    }
  };

  template<typename F_mu>
  struct mu_referenced_theta_update {
    template<typename T>
    void operator()(F_mu&& f,
                    Eigen::VectorXd& theta,
                    Eigen::VectorXd& mu,
                    Eigen::MatrixXd& sigma,
                    std::vector<Eigen::Matrix<double, -1, 1>> const& phi,
                    std::vector<T> const&& covariate) {
      mu_referenced_multi_normal_lpdf<F_mu, T> 
        lp(f, covariate, phi, sigma);

      // solve Hx = b where the rhs b is the score function
      // and H is the Hessian/information matrix
      int n_mu = mu.size();
      int n = theta.size();
      Eigen::VectorXd rhs(n);
      double lp_val;
      stan::math::gradient(lp, theta, lp_val, rhs);

      // approximate hessian/information matrix
      mu_referenced_f<F_mu, T> mu_f(f, covariate);
      Eigen::MatrixXd dmu(n_mu, n);
      Eigen::VectorXd mu_f_val(n_mu);
      jacobian(mu_f, theta, mu_f_val, dmu);

      // Precision matrix
      Eigen::MatrixXd inv_sigma = 
        sigma.llt().solve(Eigen::MatrixXd::Identity(sigma.rows(), sigma.cols()));

      Eigen::MatrixXd hessian = dmu.transpose() * inv_sigma * dmu;
      Eigen::VectorXd dtheta = hessian.llt().solve(dmu);

      double a = 1.0;
      while (!(lp(theta) < lp_val && a > 0.05)) {
        theta += a * dtheta;
        a /= 1.414;
      }
    }
  };

  template<>
  struct mu_referenced_theta_update<nullptr_t> {
    template<typename T>
    void operator()(nullptr_t&& f,
                    Eigen::VectorXd& theta,
                    Eigen::VectorXd& mu,
                    Eigen::MatrixXd& sigma,
                    std::vector<Eigen::Matrix<double, -1, 1>> const& phi,
                    std::vector<T> const&& covariate) {}
  };

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
  Eigen::VectorXd theta_;       /**< mu-referenced parameters */
 public:
  explicit welford_sa_estimator(int n)
      : welford_covar_estimator(n), iter(0),
        k1_(-1),
        curr_m_(m_), curr_m2_(m2_) {
    restart();
  }

  // explicit welford_sa_estimator(int n, int n_theta)
  //     : welford_covar_estimator(n), iter(0),
  //       k1_(-1),
  //       curr_m_(m_), curr_m2_(m2_), theta_(n_theta) {
  //   restart();
  // }

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
    if (false) {
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
    }
    for (auto i = 0; i < covar.rows(); ++i) {
      assert(covar(i, i) > 0);      
    }
  }
};

class em_pop {
 public:
  Eigen::VectorXd* p_mu;
  Eigen::VectorXd* p_theta;
  Eigen::MatrixXd* p_sigma;

  size_t n_subj;
  size_t n;
  size_t n_mu;
  welford_sa_estimator* covar_est;

  em_pop() : p_mu(nullptr), p_theta(nullptr), p_sigma(nullptr), 
             n_subj(0), n(0), n_mu(0), covar_est(nullptr)
  {}

  Eigen::VectorXd& mu() {
    return *p_mu;
  }

  Eigen::MatrixXd & sigma() {
    return *p_sigma;
  }

  Eigen::VectorXd & theta() {
    return *p_theta;
  }

  void set_mu(Eigen::VectorXd & mu) { 
    p_mu = &mu; 
    n_mu = mu.size(); 
  }

  void set_theta(Eigen::VectorXd & theta) { 
    p_theta = &theta; 
    n = theta.size();
  }

  void set_sigma(Eigen::MatrixXd & sigma) { 
    p_sigma = &sigma;
    assert(n_mu == sigma.rows());
    assert(n_mu == sigma.cols());
  }

  /** 
   * use constrained individual param sample to update population
   * sufficient statistics.
   * 
   */
  void update_hyper_param(std::vector<Eigen::VectorXd> const& phi) {
    Eigen::VectorXd& ref_mu = *p_mu;
    Eigen::VectorXd& ref_theta = *p_theta;
    Eigen::MatrixXd& ref_sigma = *p_sigma;

    n_subj = phi.size();
    for (auto i = 0; i < n_subj; ++i) {
      covar_est -> add_sample(phi[i]);
    }
    covar_est -> weighted_sample_estimate(ref_mu, ref_sigma);
  }

  void em_step() {
    covar_est -> incr_iter();
    covar_est -> restart();

    if (p_theta != nullptr) {
      std::cout << "[torsten debug print] EM mean: " << theta().transpose() << "\n";      
    } else {
      std::cout << "[torsten debug print] EM mean: " << mu().transpose() << "\n";      
    }

    std::cout << "[torsten debug print] EM covar: " << sigma() << "\n";

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

