#include <fmt/format.h>
#include <map>
#include <stdexcept>
#include <tuple>

#include <utility>
#include <xtensor/xadapt.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "ecole/instance/combinatorial-auction.hpp"
#include "ecole/scip/cons.hpp"
#include "ecole/scip/model.hpp"
#include "ecole/scip/utils.hpp"
#include "ecole/scip/var.hpp"

namespace ecole::instance {

/*******************************************
 *  CombinatorialAuctionGenerator methods  *
 *******************************************/

CombinatorialAuctionGenerator::CombinatorialAuctionGenerator(RandomEngine random_engine_, Parameters parameters_) :
	random_engine{random_engine_}, parameters{parameters_} {}
CombinatorialAuctionGenerator::CombinatorialAuctionGenerator(Parameters parameters_) :
	CombinatorialAuctionGenerator{ecole::spawn_random_engine(), parameters_} {}
CombinatorialAuctionGenerator::CombinatorialAuctionGenerator() : CombinatorialAuctionGenerator(Parameters{}) {}

scip::Model CombinatorialAuctionGenerator::next() {
	return generate_instance(random_engine, parameters);
}

void CombinatorialAuctionGenerator::seed(Seed seed) {
	random_engine.seed(seed);
}

namespace {

using std::size_t;
using vector = std::vector<size_t>;

template <typename T> using xvector = xt::xtensor<T, 1>;
template <typename T> using xmatrix = xt::xtensor<T, 2>;

/** Logs warnings for invalid bids.
 *
 * Logging warnings may be useful when using non-default parameters for the
 * generator as some sets of parameters will cause the generator to have
 * mostly invalid bids, making the compute time to generate an instance long.
 */
class Logger {
public:
	Logger(bool print_, std::string pattern_) : pattern{std::move(std::move(pattern_))}, print{print_} {}
	Logger(bool print_) : print{print_} {}
	template <typename T> void log(T&& message) {
		if (print) {
			fmt::print(pattern, message);
		}
	}

private:
	std::string pattern = "ecole::instance: {}\n";
	bool print = false;
};

/** Sample with replacement based on weights given by a probability or weight vector.
 *
 * Samples n_samples values from a weighted distribution defined by the weights.
 * The values are in the range of [1, weights.size()].
 */
auto arg_choice_without_replacement(size_t n_samples, xvector<double> weights, RandomEngine& random_engine) {

	auto const weight_sum = xt::sum(weights)();
	xvector<double> weights_cumsum = xt::cumsum(weights);

	xvector<double> samples = xt::random::rand({n_samples}, 0.0, weight_sum, random_engine);
	xvector<size_t> indices({n_samples});

	for (size_t i = 0; i < n_samples; ++i) {
		for (size_t j = 0; j < weights.size() - 1; ++j) {
			if (samples[i] < weights_cumsum[j]) {
				indices[i] = j;
				break;
			}
		}
	}

	return indices;
}

/** Choose the next item to be added to the bundle/sub-bundle.
 */
auto choose_next_item(
	const xvector<int>& bundle_mask,
	const xvector<double>& interests,
	const xmatrix<double>& compats,
	RandomEngine& random_engine) {

	auto compats_masked = xt::index_view(compats, bundle_mask);
	auto compats_masked_mean = xt::sum(compats_masked, 0);
	xvector<double> probs = (1 - bundle_mask) * interests * compats_masked_mean;
	return arg_choice_without_replacement(1, probs, random_engine)(0);
}

/** Gets price of the bundle
 */
auto get_bundle_price(const vector& bundle, const xvector<double>& private_values, bool integers, double additivity) {

	auto bundle_sum = xt::sum(xt::index_view(private_values, bundle))();
	auto bundle_power = std::pow(static_cast<double>(bundle.size()), 1.0 + additivity);
	auto price = bundle_sum + bundle_power;

	if (integers) {
		price = floor(price);
	}

	return price;
}

/** Generate initial bundle, choose first item according to bidder interests
 */
auto get_bundle(
	const xmatrix<double>& compats,
	const xvector<double>& private_interests,
	const xvector<double>& private_values,
	size_t n_items,
	bool integers,
	double additivity,
	double add_item_prob,
	RandomEngine& random_engine) {

	size_t item = arg_choice_without_replacement(1, private_interests, random_engine)(0);

	xvector<size_t> bundle_mask = xt::zeros<size_t>({n_items});
	bundle_mask(item) = 1;

	// add additional items, according to bidder interests and item compatibilities
	while (true) {
		double sampled_prob = xt::random::rand({1}, 0.0, 1.0, random_engine)[0];
		if (sampled_prob >= add_item_prob) {
			break;
		}

		if (xt::sum(bundle_mask)() == n_items) {
			break;
		}

		item = choose_next_item(bundle_mask, private_interests, compats, random_engine);
		bundle_mask(item) = 1;
	}

	vector bundle = xt::nonzero(bundle_mask)[0];

	auto price = get_bundle_price(bundle, private_values, integers, additivity);

	return std::make_tuple(bundle, price);
}

/** Genereate the set of subsititue bundles
 */
auto get_substitute_bundles(
	const vector& bundle,
	const xmatrix<double>& compats,
	const xvector<double>& private_interests,
	const xvector<double>& private_values,
	size_t n_items,
	bool integers,
	double additivity,
	RandomEngine& random_engine) {

	// get substitute bundles
	std::vector<std::tuple<vector, double>> sub_bundles{};

	for (auto item : bundle) {

		// at least one item must be shared with initial bundle
		xvector<size_t> sub_bundle_mask = xt::zeros<size_t>({n_items});
		sub_bundle_mask(item) = 1;

		// add additional items, according to bidder interests and item compatibilities
		while (true) {
			if (xt::sum(sub_bundle_mask)() >= bundle.size()) {
				break;
			}
			item = choose_next_item(sub_bundle_mask, private_interests, compats, random_engine);
			sub_bundle_mask(item) = 1;
		}

		vector sub_bundle = xt::nonzero(sub_bundle_mask)[0];

		auto sub_price = get_bundle_price(sub_bundle, private_values, integers, additivity);

		sub_bundles.emplace_back(sub_bundle, sub_price);
	}

	return sub_bundles;
}

/** Method to be used in sorting tuples of (bundle,price) by the price.
 *
 *  This method is used add bundles when adding subsitute bundles by highest price
 *  bundles first.
 */
bool sort_by_price(std::tuple<vector, double>& a, std::tuple<vector, double>& b) {
	return (std::get<1>(a) > std::get<1>(b));
}

/** Adds valid substitue bundles to bidder_bids.
 *
 * Bundles are added in descending price order until either
 * all valid bundles are added or the maximum number of bids
 * is reached.
 */
auto add_bundles(
	std::map<vector, double>* bidder_bids,
	std::vector<std::tuple<vector, double>> sub_bundles,
	const xvector<double>& values,
	const vector& bundle,
	double price,
	size_t bid_index,
	Logger logger,
	double budget_factor,
	double resale_factor,
	size_t max_n_sub_bids,
	size_t n_bids) {

	std::vector<vector> bundles{};
	std::vector<double> prices{};

	auto budget = budget_factor * price;
	auto min_resale_value = resale_factor * xt::sum(xt::index_view(values, bundle))();

	// sort for highest price substitute bundles first
	std::sort(sub_bundles.begin(), sub_bundles.end(), sort_by_price);

	// add valid substitute bundles to bidder_bids
	for (auto [sub_bundle, sub_price] : sub_bundles) {

		if ((*bidder_bids).size() >= max_n_sub_bids + 1 || bid_index + (*bidder_bids).size() >= n_bids) {
			break;
		}

		if (sub_price < 0) {
			logger.log("warning, negatively priced substitutable bundle avoided");
			continue;
		}

		if (sub_price > budget) {
			logger.log("warning, over priced substitutable bundle avoided");
			continue;
		}

		if (xt::sum(xt::index_view(values, sub_bundle))() < min_resale_value) {
			logger.log("warning, substitutable bundle below min resale value avoided");
			continue;
		}

		if ((*bidder_bids).count(sub_bundle)) {
			logger.log("warning, duplicated substitutable bundle avoided");
			continue;
		}

		(*bidder_bids)[sub_bundle] = sub_price;
	}
}

/** Adds a single variable with the coefficient price
 *
 */
auto add_var(SCIP* scip, size_t i, double price) {
	auto const name = fmt::format("x_{}", i);
	auto unique_var = scip::create_var_basic(scip, name.c_str(), 0., 1., price, SCIP_VARTYPE_BINARY);
	auto* var_ptr = unique_var.get();
	scip::call(SCIPaddVar, scip, var_ptr);
	return var_ptr;
}

/** Adds all constraints to the SCIP model.
 *
 */
auto add_constraints(SCIP* scip, xvector<SCIP_VAR*> vars, const std::vector<vector>& bids_per_item) {
	size_t index = 0;
	for (auto item_bids : bids_per_item) {
		if (!item_bids.empty()) {
			auto cons_vars = xvector<SCIP_VAR*>{{item_bids.size()}};
			auto coefs = xvector<scip::real>(cons_vars.shape(), 1.);
			for (size_t j = 0; j < item_bids.size(); ++j) {
				cons_vars(j) = vars(item_bids[j]);
			}
			auto name = fmt::format("c_{}", index);
			auto const inf = SCIPinfinity(scip);
			auto cons =
				scip::create_cons_basic_linear(scip, name.c_str(), cons_vars.size(), &cons_vars(0), coefs.data(), -inf, 1.0);
			scip::call(SCIPaddCons, scip, cons.get());
		}
		++index;
	}
}

}  // namespace

/******************************************************
 *  CombinatorialAuctionGenerator::generate_instance  *
 ******************************************************/

scip::Model CombinatorialAuctionGenerator::generate_instance(RandomEngine& random_engine, Parameters parameters) {

	// check that parameters are valid
	if (!(parameters.min_value >= 0 && parameters.max_value >= parameters.min_value)) {
		throw std::invalid_argument(
			"Parameters max_value and min_value must be defined such that: 0 <= min_value <= max_value.");
	}

	if (!(parameters.add_item_prob >= 0 && parameters.add_item_prob <= 1)) {
		throw std::invalid_argument("Parameter add_item_prob must be in range [0,1].");
	}

	// initialize logger for warnings
	auto logger = Logger(parameters.warnings);

	// get values
	xvector<double> rand_val = xt::random::rand({parameters.n_items}, 0.0, 1.0, random_engine);
	xvector<double> values = parameters.min_value + (parameters.max_value - parameters.min_value) * rand_val;

	// get compatibilities
	xmatrix<double> compats_rand = xt::random::rand({parameters.n_items, parameters.n_items}, 0.0, 1.0, random_engine);
	xmatrix<double> compats = xt::triu(compats_rand, 1);
	compats = compats + xt::transpose(compats);
	compats = compats / xt::sum(compats, 1);

	size_t n_dummy_items = 0;
	size_t bid_index = 0;
	std::vector<std::tuple<vector, double>> bids{parameters.n_bids};

	while (bid_index < parameters.n_bids) {

		// bidder item values (buy price) and interests
		xvector<double> private_interests = xt::random::rand({parameters.n_items}, 0.0, 1.0, random_engine);
		xvector<double> private_values = values +
																		 parameters.max_value * parameters.value_deviation * (2 * private_interests - 1);

		// substitutable bids of this bidder
		std::map<vector, double> bidder_bids = {};

		// generate initial bundle, choose first item according to bidder interests
		auto [bundle, price] = get_bundle(
			compats,
			private_interests,
			private_values,
			parameters.n_items,
			parameters.integers,
			parameters.additivity,
			parameters.add_item_prob,
			random_engine);

		// restart bid if price < 0
		if (price < 0) {
			logger.log("warning, negatively priced bundle avoided");
			continue;
		}

		// add bid to bidder_bids
		bidder_bids[bundle] = price;

		// get substitute bundles
		auto substitute_bundles = get_substitute_bundles(
			bundle,
			compats,
			private_interests,
			private_values,
			parameters.n_items,
			parameters.integers,
			parameters.additivity,
			random_engine);

		// add XOR bundles to bidder_bids
		add_bundles(
			&bidder_bids,
			substitute_bundles,
			values,
			bundle,
			price,
			bid_index,
			logger,
			parameters.budget_factor,
			parameters.resale_factor,
			parameters.max_n_sub_bids,
			parameters.n_bids);

		// get dummy item if required
		size_t dummy_item = 0;
		if (bidder_bids.size() > 2) {
			dummy_item = parameters.n_items + n_dummy_items;
			++n_dummy_items;
		}

		for (auto const& [b, p] : bidder_bids) {
			auto bund_copy = b;
			if (dummy_item) {
				bund_copy.push_back(dummy_item);
			}
			bids[bid_index] = std::make_tuple(bund_copy, p);

			++bid_index;
		}
	}  // loop to get bids

	// create scip model
	auto model = scip::Model::prob_basic();
	auto* const scip = model.get_scip_ptr();
	scip::call(SCIPsetObjsense, scip, SCIP_OBJSENSE_MAXIMIZE);

	//  initialize bids_per_item
	std::vector<vector> bids_per_item{parameters.n_items + n_dummy_items};
	for (size_t i = 0; i < parameters.n_items + n_dummy_items; ++i) {
		vector bids_per_item_i{};
		bids_per_item.push_back(bids_per_item_i);
	}

	// get bids per item
	size_t i = 0;
	for (auto [bundle, price] : bids) {
		for (unsigned long j : bundle) {
			bids_per_item[j].push_back(i);
		}
		++i;
	}

	// add variables
	i = 0;
	auto vars = xvector<SCIP_VAR*>{{bids.size()}};
	for (auto [_, price] : bids) {
		vars(i) = add_var(scip, i, price);
		++i;
	}

	// add constraints
	add_constraints(scip, vars, bids_per_item);

	return model;
}

}  // namespace ecole::instance
