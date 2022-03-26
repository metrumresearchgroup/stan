#include <stan/model/em_pop.hpp>
#include <gtest/gtest.h>
#include <stan/io/empty_var_context.hpp>
#include <test/test-models/good/optimization/rosenbrock.hpp>
#include <stan/callbacks/stream_writer.hpp>

Eigen::MatrixXd data_outer(Eigen::MatrixXd const& x) {
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(x.cols(), x.cols());
  for (auto i = 0; i < x.rows(); ++i) {
    res += x.row(i).transpose() * x.row(i);
  }
  return res;
}

TEST(Services, welford_stochastic_approx) {
  int k = 2;
  int n = 4;
  int ndata = k + 5;
  int ns = 5;
  stan::model::welford_sa_estimator est(n, k);
  
  // data from R: 
  // sigma <- matrix(as.vector(sigma),n)
  // d <- MASS::mvrnorm(ns, rep(0,4), sigma)
  std::vector<Eigen::MatrixXd> data(ndata);
  data[0] = Eigen::MatrixXd(ns, n);
  data[0] << 
-3.536991452,  3.012750409, -4.186368321, -1.383508151, -5.934603406,
 3.221494630, -4.910399483, -3.543404006,  2.045287354,  1.949278946,
 0.766677237,  2.185374538,  2.863613290, -3.587371064,  4.732582036,
 0.130604376,  0.142331046, -1.698912347, -2.446560601,  0.068174549;

  data[1] = Eigen::MatrixXd(ns, n);
  data[1] <<
 4.3154684, -4.1143153, -4.3404136,  3.9671535,  0.2562683,  0.1476272,
 1.4256616, -0.2015520, -1.9578759, -0.5585525,  1.1948528, -2.5994397,
-0.8991225, -0.9861194, -5.9740846,  0.6963397, -0.5383476, -0.9467984,
-1.2475388, -0.3029859;

  data[2] = Eigen::MatrixXd(ns, n);
  data[2] <<
 0.223535720,  0.522368820,  0.110600203,  0.502072943,  4.096646991,
-0.539064219,  3.904205981,  2.597555492, -0.445272446,  2.247143393,
-0.223055860,  0.710786801,  6.803877093, -0.354414659,  4.148987701,
 5.282974041, -0.007912912, -0.810107510,  1.045093547, -0.390761483;


  data[3] = Eigen::MatrixXd(ns, n);
  data[3] <<
  1.1616459, -2.38452961,  4.9719019, -1.21208203,
 -0.2029260, -0.08910776, -0.6673973, -0.05358716,
  2.5730691,  0.67506973,  2.3011144,  2.00218103,
 -1.9015477, -1.52890683, -3.0251583, -1.50877056,
 -2.1288614, -1.03368662, -0.7584694, -1.96920243;

  data[4] = Eigen::MatrixXd(ns, n);
  data[4] <<
 -3.1829432, -1.36240863,  1.7338691, -3.83868000,
  2.3092547,  4.12948046,  3.5872807,  2.72247964,
 -1.1986597, -2.31120168, -1.7316614, -1.32097794,
  3.3633133, -0.86462422,  3.2353655,  1.99830487,
  0.1718646,  0.27761092, -1.2092768,  0.75956002;

  data[5] = Eigen::MatrixXd(ns, n);
  data[5] <<
  0.22537544, -0.5445841, 2.561873, -0.9306133,
  0.34174169, -1.5206117, 2.921336, -1.1276736,
 -0.08767352, -0.5849031, 1.767479, -0.8640164,
 -2.83513466,  0.8947674, 1.729721, -2.6163908,
  5.51868261,  0.9906226, 7.259822,  3.6141345;

  data[6] = Eigen::MatrixXd(ns, n);
  data[6] <<
  4.2448092, -1.1130604,  2.291605,  2.8553310,
 -1.1580222,  0.2963468,  1.645210, -1.2705188,
 -0.4021544,  4.3971633, -4.002000,  2.1319032,
 -0.2613446,  3.5438395, -4.074165,  2.1143073,
  3.9169277, -4.5094438,  7.233047,  0.3210838;

  EXPECT_EQ(est.k1(), k);

  Eigen::VectorXd mu(n), mu_alt(n), mu0(n), mu1(n), mu2(n), mu3(n);
  Eigen::MatrixXd sigma(n, n), sigma_alt(n, n), sigma0(n, n),
    sigma1(n, n), sigma2(n, n), sigma3(n, n);

  for (auto iter = 0; iter < ndata; ++iter) {
    for (auto i = 0; i < data[iter].rows(); ++i) {
      est.add_sample(data[iter].row(i));
    }
    est.sample_mean(mu0);
    est.sample_covariance(sigma0);
    est.weighted_sample_estimate(mu, sigma);
    if (iter <= k + 1) {
      for (auto i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(mu0(i), mu(i));
        for (auto j = 0; j < n; ++j) {
          EXPECT_FLOAT_EQ(sigma0(j, i), sigma(j, i));
        }
      }
    } else {
      if (iter == k + 2) {
        mu_alt = 0.5 * mu0 + 0.5 * mu1;
        sigma_alt = (0.5 * data_outer(data[iter]) + 0.5 * data_outer(data[iter-1]))/(ns-1);
        sigma_alt -= mu_alt * mu_alt.transpose() * ns/(ns-1);
      } else if (iter == k + 3) {
        mu_alt = 0.33333333 * mu0 + 0.66666667 * (0.5 * mu1 + 0.5 * mu2);
        sigma_alt = (0.33333333 * data_outer(data[iter]) + 
                     0.66666667 * (0.5 * data_outer(data[iter-1]) + 0.5 * data_outer(data[iter-2])))/(ns-1);
        sigma_alt -= mu_alt * mu_alt.transpose() * ns/(ns-1);
      } else if (iter == k + 4) {
        mu_alt = 0.25 * mu0 + 0.75 * (0.33333333 * mu1 + 0.66666667 * (0.5 * mu2 + 0.5 * mu3));
        sigma_alt = (0.25 * data_outer(data[iter]) + 0.75 * (0.33333333 * data_outer(data[iter-1]) + 
                     0.66666667 * (0.5 * data_outer(data[iter-2]) + 0.5 * data_outer(data[iter-3]))))/(ns-1);
        sigma_alt -= mu_alt * mu_alt.transpose() * ns/(ns-1);
      }
      for (auto i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(mu_alt(i), mu(i));
        for (auto j = 0; j < n; ++j) {
          EXPECT_FLOAT_EQ(sigma_alt(j, i), sigma(j, i));
        }
      }
    }

    mu3 = mu2;                  // from data curr_iter - 3
    mu2 = mu1;                  // from data curr_iter - 2
    mu1 = mu0;                  // from data curr_iter - 1

    est.incr_iter();
    est.restart();
  }
}
