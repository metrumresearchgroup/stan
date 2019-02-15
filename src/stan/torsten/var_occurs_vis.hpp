#ifndef STAN_LANG_TORSTEN_AST_FUN_VAR_OCCURS_VIS_HPP
#define STAN_LANG_TORSTEN_AST_FUN_VAR_OCCURS_VIS_HPP

bool operator()(const univariate_integral_control& e) const;
bool operator()(const generalOdeModel_control& e) const;
bool operator()(const generalOdeModel& e) const;
bool operator()(const pop_pk_generalOdeModel& e) const;

#endif
