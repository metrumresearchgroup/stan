#ifndef STAN_MODEL_MODEL_EM_POP_HPP
#define STAN_MODEL_MODEL_EM_POP_HPP

#include <stan/io/var_context.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/fun/welford_var_estimator.hpp>
#include <stan/math/prim/fun/welford_covar_estimator.hpp>
#include <stan/math/rev/functor/gradient.hpp>
#include <stan/math/rev/functor/jacobian.hpp>
#include <stan/math/prim/prob/multi_normal_lpdf.hpp>
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

    template<typename T,
             require_vector_like_t<T>* = nullptr>
    return_type_t<T> operator()(T const& theta) const {
      return_type_t<T> l = 0.0;
      size_t n_subj = subj_sample.size();
      for (auto i = 0; i < n_subj; ++i) {
        l += stan::math::multi_normal_lpdf(subj_sample[i], f(theta, covariate[i], nullptr), sigma);
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

    template<typename T,
             stan::require_vector_like_t<T>* = nullptr>
    Eigen::Matrix<stan::return_type_t<T>, -1, 1> 
    operator()(T const& theta) const {
      return f(theta, covariate, nullptr);
    }
  };

  /** 
   * Calculate score function wrt theta. We make assumption that theta
   * influence yhat only through mu, thus the score calculation
   * doesn't involve yhat, or phi -> yhat mapping, e.g. compartment models.
   * We also assume multi-normal: phi ~ N(mu(theta), sigma)
   * 
   * @param mu mu-referencing function. theta -> phi
   * @param theta population mean
   * @param sigma covariance matrix
   * @param phi individual parameters
   * @param covariate covariates
   * @param score d(log(f1 * f2))/dtheta
   */
  template<typename F_mu, typename T>
  void accumulate_score(F_mu&& mu,
                        Eigen::VectorXd& theta,
                        Eigen::MatrixXd& sigma,
                        std::vector<Eigen::Matrix<double, -1, 1>> const& phi,
                        std::vector<T> const& covariate,
                        Eigen::VectorXd & score) {
    mu_referenced_multi_normal_lpdf<F_mu, T> lp(mu, covariate, phi, sigma);
    double lp_val;
    stan::math::gradient(lp, theta, lp_val, score);
  }

  template<typename F_mu, typename T>
  void solve_theta_from_hessian(F_mu&& mu,
                                Eigen::VectorXd& theta,
                                Eigen::MatrixXd const& sigma,
                                Eigen::MatrixXd const& inv_sigma,
                                std::vector<Eigen::Matrix<double, -1, 1>> const& phi,
                                std::vector<T> const& covariate,
                                Eigen::VectorXd & score,
                                Eigen::MatrixXd & hessian) {
    Eigen::MatrixXd dmu;
    Eigen::VectorXd mu_f_val;
    for (auto i = 1; i < covariate.size(); ++i) {
      mu_referenced_f<F_mu, T> mu_f(mu, covariate[i]);
      stan::math::jacobian(mu_f, theta, mu_f_val, dmu);
      if (i == 0) {
        hessian = dmu.transpose() * inv_sigma * dmu;
      } else {
        hessian += dmu.transpose() * inv_sigma * dmu;
      }
    }
    Eigen::MatrixXd h = hessian + hessian.transpose();
    h = h.array() * 0.5;

    std::cout << "taki test H: " <<  "\n";
    std::cout << h << "\n";
    Eigen::VectorXd dtheta = h.fullPivLu().solve(score);
    std::cout << "taki test inv_h*dtheta: " << h * dtheta - score << "\n";
    double a = 1.0;
    mu_referenced_multi_normal_lpdf<F_mu, T> lp(mu, covariate, phi, sigma);
    std::cout << "[torsten debug print] test before: " << lp(theta) << "\n";
    // while (lp(theta + a * dtheta) < lp(theta) && a > 0.1) {
    //   a /= 1.414;
    // }
    Eigen::VectorXd theta_next = theta + a * dtheta;
    // if (lp(theta_next) > lp(theta)) {
      theta  = theta_next;
    // }
    std::cout << "[torsten debug print] test after: " << a << " " << lp(theta) << "\n";
  }

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
  Eigen::VectorXd score;
  Eigen::MatrixXd hessian;
  Eigen::MatrixXd inv_sigma;

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
    if (p_mu == nullptr) {
      n_mu = sigma.rows();
    } else {
      assert(n_mu == sigma.rows());
      assert(n_mu == sigma.cols());
    }
  }

  void set_workspace() {
    score.resize(n);
    hessian.resize(n, n);
    inv_sigma.resize(n_mu, n_mu);
  }

  /** 
   * use constrained individual param sample to update population
   * sufficient statistics.
   * 
   */
  void accumulate_sample(std::vector<Eigen::VectorXd> const& phi) {
    n_subj = phi.size();
    for (auto i = 0; i < n_subj; ++i) {
      covar_est -> add_sample(phi[i]);
    }
  }

  void set_inv_sigma() {
    Eigen::MatrixXd& ref_sigma = *p_sigma;
    inv_sigma = ref_sigma.colPivHouseholderQr().solve(Eigen::MatrixXd::Identity(n_mu, n_mu));
    std::cout << "taki test inv_sigma: " << "\n";
    std::cout << inv_sigma  << "\n";
  }

  /** 
   * Update theta or mu & sigma according EM iteration
   * 
   * The algorithm flow:
   * 1. The actual model class impl add_samle() that
   *    retrieves constrained samples "phi"
   * mu_referenced_hessian<F_mu> object to update hessian
   * The callback in the actual model is in the update_theta function
   *
   * void update_theta(Eigen::VectorXd & theta) {
   *
   * }
   * 
   * 
   */  
  void em_step() {
    Eigen::VectorXd& ref_mu = *p_mu;
    Eigen::VectorXd& ref_theta = *p_theta;
    Eigen::MatrixXd& ref_sigma = *p_sigma;

    if (p_theta != nullptr) {
      // update_theta(ref_theta);
    } else {
      covar_est -> weighted_sample_estimate(ref_mu, ref_sigma);
      covar_est -> incr_iter();
      covar_est -> restart();
    }

    // prepare next EM iteration

    // Precision matrix
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

