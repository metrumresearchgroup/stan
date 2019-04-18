#include <gtest/gtest.h>
#include <test/unit/lang/utility.hpp>

TEST(lang_parser, univariate_integrate_good) {
  test_parsable("torsten/univariate_integral_good");
}
TEST(lang_parser, univariate_integrate_rk45_bad) {
  test_throws("torsten/univariate_integral/bad_fun_type_rk45",
              "first argument to univariate_integral_rk45 "
              "must be the name of a function with signature");
  test_throws("torsten/univariate_integral/bad_t0_rk45",
              "second argument to univariate_integral_rk45 "
              "must have type real for time limit;");
  test_throws("torsten/univariate_integral/bad_t1_rk45",
              "third argument to univariate_integral_rk45 "
              "must have type real for time limit;");
  test_throws("torsten/univariate_integral/bad_theta_rk45",
              "fourth argument to univariate_integral_rk45 "
              "must have type real[] for parameters;");
  test_throws("torsten/univariate_integral/bad_x_r_rk45",
              "fifth argument to univariate_integral_rk45 "
              "must have type real[] for real data;");
  test_throws("torsten/univariate_integral/bad_x_i_rk45",
              "sixth argument to univariate_integral_rk45 "
              "must have type int[] for integer data;");
}
TEST(lang_parser, univariate_integrate_bdf_bad) {
  test_throws("torsten/univariate_integral/bad_fun_type_bdf",
              "first argument to univariate_integral_bdf "
              "must be the name of a function with signature");
  test_throws("torsten/univariate_integral/bad_t0_bdf",
              "second argument to univariate_integral_bdf "
              "must have type real for time limit;");
  test_throws("torsten/univariate_integral/bad_t1_bdf",
              "third argument to univariate_integral_bdf "
              "must have type real for time limit;");
  test_throws("torsten/univariate_integral/bad_theta_bdf",
              "fourth argument to univariate_integral_bdf "
              "must have type real[] for parameters;");
  test_throws("torsten/univariate_integral/bad_x_r_bdf",
              "fifth argument to univariate_integral_bdf "
              "must have type real[] for real data;");
  test_throws("torsten/univariate_integral/bad_x_i_bdf",
              "sixth argument to univariate_integral_bdf "
              "must have type int[] for integer data;");
}

TEST(lang_parser, generalOdeModel_good) {
  test_parsable("torsten/generalOdeModel_good");
}

TEST(lang_parser, mixOdeModel_good) {
  test_parsable("torsten/mixOdeModel_good");
}

/*****************************************************************
 pmx_solve
 ****************************************************************/
TEST(lang_parser, generalCptModel) {
  test_parsable("torsten/generalCptModel");
  test_throws("torsten/pmx_solve/rk45_bad_functor", "1st argument to pmx_solve_rk45 must be a function with signature (real, real[], real[], real[], int[]) : real[]");
  test_throws("torsten/pmx_solve/rk45_bad_time"   , "3rd argument to pmx_solve_rk45 must be type real[] for time;");
  test_throws("torsten/pmx_solve/rk45_bad_amt"    , "4th argument to pmx_solve_rk45 must be type real[] for amount;");
  test_throws("torsten/pmx_solve/rk45_bad_rate"   , "5th argument to pmx_solve_rk45 must be type real[] for rate;");
  test_throws("torsten/pmx_solve/rk45_bad_ii"     , "6th argument to pmx_solve_rk45 must be type real[] for inter-dose interval;");
  test_throws("torsten/pmx_solve/rk45_bad_evid"   , "7th argument to pmx_solve_rk45 must be type int[] for event ID;");
  test_throws("torsten/pmx_solve/rk45_bad_cmt"    , "8th argument to pmx_solve_rk45 must be type int[] for compartment ID;");
  test_throws("torsten/pmx_solve/rk45_bad_addl"   , "9th argument to pmx_solve_rk45 must be type int[] for number of additional doses;");
  test_throws("torsten/pmx_solve/rk45_bad_ss"     , "10th argument to pmx_solve_rk45 must be type int[] for steady state flags;");
  test_throws("torsten/pmx_solve/rk45_bad_param"  , "11th argument to pmx_solve_rk45 must be type real[] or real[ , ] for ODE parameters;");
  test_throws("torsten/pmx_solve/rk45_bad_biovar" , "12th argument to pmx_solve_rk45 must be type real[] or real[ , ] for bioavailability;");
  test_throws("torsten/pmx_solve/rk45_bad_tlag"   , "13th argument to pmx_solve_rk45 must be type real[] or real[ , ] for lag times;");
  test_throws("torsten/pmx_solve/rk45_var_rtol"   , "14th argument to pmx_solve_rk45 for relative tolerance must be data only");
}

/*****************************************************************
 pmx_solve_group
 ****************************************************************/
TEST(lang_parser, pmx_solve_group) {
  test_parsable("torsten/pmx_solve_group");
  test_throws("torsten/pmx_solve_group/rk45_bad_functor", "1st argument to pmx_solve_group_rk45 must be a function with signature (real, real[], real[], real[], int[]) : real[]");
  test_throws("torsten/pmx_solve_group/rk45_bad_time"   , "4th argument to pmx_solve_group_rk45 must be type real[] for time;");
  test_throws("torsten/pmx_solve_group/rk45_bad_amt"    , "5th argument to pmx_solve_group_rk45 must be type real[] for amount;");
  test_throws("torsten/pmx_solve_group/rk45_bad_rate"   , "6th argument to pmx_solve_group_rk45 must be type real[] for rate;");
  test_throws("torsten/pmx_solve_group/rk45_bad_ii"     , "7th argument to pmx_solve_group_rk45 must be type real[] for inter-dose interval;");
  test_throws("torsten/pmx_solve_group/rk45_bad_evid"   , "8th argument to pmx_solve_group_rk45 must be type int[] for event ID;");
  test_throws("torsten/pmx_solve_group/rk45_bad_cmt"    , "9th argument to pmx_solve_group_rk45 must be type int[] for compartment ID;");
  test_throws("torsten/pmx_solve_group/rk45_bad_addl"   , "10th argument to pmx_solve_group_rk45 must be type int[] for number of additional doses;");
  test_throws("torsten/pmx_solve_group/rk45_bad_ss"     , "11th argument to pmx_solve_group_rk45 must be type int[] for steady state flags;");
  test_throws("torsten/pmx_solve_group/rk45_bad_param"  , "12th argument to pmx_solve_group_rk45 must be type real[] or real[ , ] for ODE parameters;");
  test_throws("torsten/pmx_solve_group/rk45_bad_biovar" , "13th argument to pmx_solve_group_rk45 must be type real[] or real[ , ] for bioavailability;");
  test_throws("torsten/pmx_solve_group/rk45_bad_tlag"   , "14th argument to pmx_solve_group_rk45 must be type real[] or real[ , ] for lag times;");
  test_throws("torsten/pmx_solve_group/rk45_var_rtol"   , "15th argument to pmx_solve_group_rk45 for relative tolerance must be data only");
}

/*****************************************************************
 pmx_solve_onecpt
 ****************************************************************/
TEST(lang_parser, PKModelOneCpt_function_signatures) {
    test_parsable("torsten/PKModelOneCpt");
}

/*****************************************************************
 pmx_solve_twocpt
 ****************************************************************/
TEST(lang_parser, PKModelTwoCpt_function_signatures) {
    test_parsable("torsten/PKModelTwoCpt");
}

/*****************************************************************
 pmx_solve_linode
 ****************************************************************/
TEST(lang_parser, linOdeModel_function_signatures) {
    test_parsable("torsten/linOdeModel");
}

/*****************************************************************
 pmx_solve_onecpt_ode
 ****************************************************************/
TEST(lang_parser, mixOde1CptModel_function_signatures) {
    test_parsable("torsten/mixOde1CptModel");
}

/*****************************************************************
 pmx_solve_twocpt_ode
 ****************************************************************/
TEST(lang_parser, mixOde2CptModel_function_signatures) {
    test_parsable("torsten/mixOde2CptModel");
}
