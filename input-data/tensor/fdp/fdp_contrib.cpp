// Path attributions (Saabas) for a ranger probability forest.
//
// Each split a cell passes through credits the change in node value to the
// split's variable, so per cell: baseline + sum(contributions) = prediction.
// Splits follow ranger's Tree::predict: ordered variables go left when
// value <= split value; unordered factors (respect.unordered.factors =
// "partition") go left when the factor's bit is clear in the split value.

#include <Rcpp.h>
#include <cmath>
#include <cstdint>
using namespace Rcpp;

namespace {

struct Tree {
  IntegerVector left, right, var;
  NumericVector split;
};

Tree tree_at(const List& left, const List& right, const List& var, const List& split, int t) {
  return {left[t], right[t], var[t], split[t]};
}

inline bool is_leaf(const Tree& tree, int node) {
  return tree.left[node] == 0 && tree.right[node] == 0;
}

inline int next_node(const Tree& tree, int node, double x, bool ordered) {
  if (ordered) return x <= tree.split[node] ? tree.left[node] : tree.right[node];
  uint64_t factor = static_cast<uint64_t>(std::floor(x)) - 1;
  uint64_t bits = static_cast<uint64_t>(std::floor(tree.split[node]));
  return (bits & (1ULL << factor)) ? tree.right[node] : tree.left[node];
}

}  // namespace

// In-bag mean of y at every node of tree t, the value ranger stores for its
// leaves and the one the attribution differences.
// [[Rcpp::export]]
NumericVector fdp_node_values(List left, List right, List var, List split, int t,
                              LogicalVector ordered, NumericMatrix x, NumericVector y,
                              IntegerVector inbag) {
  Tree tree = tree_at(left, right, var, split, t);
  int nodes = tree.left.size(), n = x.nrow();
  std::vector<double> sum(nodes, 0.0), weight(nodes, 0.0);
  for (int i = 0; i < n; ++i) {
    double w = inbag[i];
    if (w == 0) continue;
    int node = 0;
    while (true) {
      sum[node] += w * y[i];
      weight[node] += w;
      if (is_leaf(tree, node)) break;
      int v = tree.var[node];
      node = next_node(tree, node, x(i, v), ordered[v]);
    }
  }
  NumericVector out(nodes, NA_REAL);
  for (int k = 0; k < nodes; ++k)
    if (weight[k] > 0) out[k] = sum[k] / weight[k];
  return out;
}

// Per-cell contributions averaged over trees, one column per variable, and the
// forest prediction in a final column.
// [[Rcpp::export]]
NumericMatrix fdp_contributions(List left, List right, List var, List split, List value,
                                LogicalVector ordered, NumericMatrix x) {
  int n = x.nrow(), p = x.ncol(), trees = left.size();
  NumericMatrix out(n, p + 1);
  for (int t = 0; t < trees; ++t) {
    Tree tree = tree_at(left, right, var, split, t);
    NumericVector val = value[t];
    for (int i = 0; i < n; ++i) {
      int node = 0;
      while (!is_leaf(tree, node)) {
        int v = tree.var[node];
        int child = next_node(tree, node, x(i, v), ordered[v]);
        out(i, v) += val[child] - val[node];
        node = child;
      }
      out(i, p) += val[node];
    }
  }
  for (int k = 0; k < n * (p + 1); ++k) out[k] /= trees;
  return out;
}
