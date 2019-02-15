#ifndef STAN_LANG_TORSTEN_GRAMMARS_SEMANTIC_ACTIONS_HPP
#define STAN_LANG_TORSTEN_GRAMMARS_SEMANTIC_ACTIONS_HPP

// called from: term_grammar
struct validate_univariate_integral_control
  : public phoenix_functor_quaternary {
  void operator()(const univariate_integral_control& ode_fun,
                  const variable_map& var_map, bool& pass,
                  std::ostream& error_msgs) const;
};
extern boost::phoenix::function<validate_univariate_integral_control>
validate_univariate_integral_control_f;

// called from: term_grammar
struct validate_generalOdeModel_control
  : public phoenix_functor_quaternary {
  void operator()(const generalOdeModel_control& ode_fun,
                  const variable_map& var_map, bool& pass,
                  std::ostream& error_msgs) const;
};
extern boost::phoenix::function<validate_generalOdeModel_control>
validate_generalOdeModel_control_f;

// called from: term_grammar
struct validate_generalOdeModel
  : public phoenix_functor_quaternary {
  void operator()(const generalOdeModel& ode_fun,
                  const variable_map& var_map, bool& pass,
                  std::ostream& error_msgs) const;
};
extern boost::phoenix::function<validate_generalOdeModel>
validate_generalOdeModel_f;

// called from: term_grammar
struct validate_pop_pk_generalOdeModel
  : public phoenix_functor_quaternary {
  void operator()(const pop_pk_generalOdeModel& ode_fun,
                  const variable_map& var_map, bool& pass,
                  std::ostream& error_msgs) const;
};
extern boost::phoenix::function<validate_pop_pk_generalOdeModel>
validate_pop_pk_generalOdeModel_f;


#endif
