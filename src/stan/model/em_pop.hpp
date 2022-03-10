#ifndef STAN_MODEL_MODEL_EM_POP_HPP
#define STAN_MODEL_MODEL_EM_POP_HPP

#include <stan/io/var_context.hpp>
#include <stan/math/rev/core.hpp>
#include <boost/random/additive_combine.hpp>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace stan {
namespace model {

class em_pop {
 public:
  Eigen::VectorXd* p_mu;
  Eigen::MatrixXd* p_sigma;
  size_t n_sample;
  size_t n_subj;
  size_t n_mu;
  size_t cov_cols;
  size_t cov_rows;
  Eigen::VectorXd theta_work;
  Eigen::VectorXd mu_all;
  Eigen::MatrixXd sigma_all;
  Eigen::VectorXd res_all;

  em_pop() : p_mu(nullptr), p_sigma(nullptr),
             n_sample(0), n_subj(0)
  {}

  Eigen::VectorXd& em_population_mean() {
    return *p_mu;
  }

  Eigen::MatrixXd & em_population_cov() {
    return *p_sigma;
  }

  /** 
   * use constrained individual param sample to update population
   * sufficient statistics
   * 
   * @param p vector consisting of scalar individual parameters
   */
  void accumulate_ind_param(Eigen::VectorXd const& p) { // 
    Eigen::VectorXd& ref_mu = *p_mu;
    Eigen::MatrixXd& ref_sigma = *p_sigma;

    n_subj = p.size();

    if (n_sample == 0) {
      theta_work = Eigen::VectorXd::Zero(n_subj);
    }

    theta_work += p;
    n_sample++;
  }

  // void accumulate_ind_param(std::vector<Eigen::VectorXd> const& p) { // 
  //   p_mu += p;
  //   p_sigma += p * p.transpose();
  //   n_sample++;
  // }

  void em_step(size_t i) {
    Eigen::VectorXd& ref_mu = *p_mu;
    Eigen::MatrixXd& ref_sigma = *p_sigma;

    theta_work /= n_sample;
    mu_all(0) += gamma(i) * theta_work.sum();
    sigma_all(0) += gamma(i) * theta_work.squaredNorm();

    ref_mu = mu_all/n_subj;
    // ref_sigma(0) = sqrt(sigma_all(0)/n_subj - ref_mu(0) * ref_mu.transpose());
    ref_sigma(0) = sqrt(sigma_all(0)/n_subj - ref_mu(0) * ref_mu(0));

    // prepare next iteration
    mu_all *= (1.0 - gamma(i + 1));
    sigma_all *= (1.0 - gamma(i + 1));

    // reset
    n_sample = 0;
  }

  double gamma(int i) {
    return (i > 50 ? 1.0/(i - 50) : 1.0);
  }
};
}
}

  // inline void add_sample(Eigen::VectorXd const & params_r) {
  //   using local_scalar_t__ = double;
  //   Eigen::VectorXd params_r__ = params_r;
  //   Eigen::Matrix<int, -1, 1> params_i__;
  //   stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
  //   static constexpr bool propto__ = true;
  //   (void) propto__;
  //   double lp__ = 0.0;
  //   (void) lp__;  // dummy to suppress unused var warning
  //   int current_statement__ = 0; 
  //   stan::math::accumulator<double> lp_accum__;
  //   local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
  //   constexpr bool jacobian__ = false;
  //   (void) DUMMY_VAR__;  // suppress unused var warning
  //   static constexpr const char* function__ = "em_eight_schools_model_namespace::write_array";
  //   (void) function__;  // suppress unused var warning
    
  //   try {
  //     Eigen::Matrix<double, -1, 1> theta =
  //        Eigen::Matrix<double, -1, 1>::Constant(n,
  //          std::numeric_limits<double>::quiet_NaN());
  //     current_statement__ = 1;
  //     theta = in__.template read<Eigen::Matrix<local_scalar_t__, -1, 1>>(n);
  //     this -> accumulate_ind_param(theta);
  //   } catch (const std::exception& e) {
  //     stan::lang::rethrow_located(e, locations_array__[current_statement__]);
  //   }
  //   }

#endif

