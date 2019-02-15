#ifndef STAN_LANG_TORSTEN_AST_FUN_IS_NIL_VIS_HPP
#define STAN_LANG_TORSTEN_AST_FUN_IS_NIL_VIS_HPP

bool operator()(const univariate_integral_control& x) const;
bool operator()(const generalOdeModel_control& x) const;
bool operator()(const generalOdeModel& x) const;
bool operator()(const pop_pk_generalOdeModel& x) const;

#endif
