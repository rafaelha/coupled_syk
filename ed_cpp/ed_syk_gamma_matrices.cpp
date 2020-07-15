#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/KroneckerProduct>
#include <array>
#include <vector>
#include <cmath>
#include <chrono>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <random>
#include <fstream>

using namespace std;
using namespace std::chrono;
using namespace Eigen;
using namespace Spectra;

typedef SparseMatrix<complex<double>> sm;
vector<sm> s;

const int N = 32; // Majorana fermions in total
const double J = 1.0;
const int num_evals = 12;
const int dimSYK = (1 << N / 4);
double eta;
double mu;

int pp[N / 2][N / 4];
const int g_dim = 1 << N / 4;
const int G_dim = 1 << N / 2;
vector<sm> g;
sm g5(g_dim, g_dim);
sm id(g_dim, g_dim);
sm H(g_dim, g_dim);
sm h(g_dim, g_dim);
sm Hint(G_dim, G_dim);
sm HLR(G_dim, G_dim);
VectorXcd gs;
VectorXd evalues;
VectorXd ev_syk;
VectorXd ev_syk_test;
MatrixXcd vecs_syk;

MatrixXcd overlap_data;

steady_clock::time_point t0;
steady_clock::time_point t1;
const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");

void define_pauli()
{
	sm sm0(2, 2);
	sm sm1(2, 2);
	sm sm2(2, 2);
	sm sm3(2, 2);
	sm0.setIdentity();
	sm1.insert(0, 1) = 1;
	sm1.insert(1, 0) = 1;
	sm2.insert(0, 1) = -1i;
	sm2.insert(1, 0) = 1i;
	sm3.insert(0, 0) = 1;
	sm3.insert(1, 1) = -1;
	s.push_back(sm0);
	s.push_back(sm1);
	s.push_back(sm2);
	s.push_back(sm3);
	//cout << s[0] << endl << s[1] << endl << s[2] << endl << s[3] << endl;
}
template<typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in)
{

	std::vector<double> linspaced;

	double start = static_cast<double>(start_in);
	double end = static_cast<double>(end_in);
	double num = static_cast<double>(num_in);

	if (num == 0) { return linspaced; }
	if (num == 1)
	{
		linspaced.push_back(start);
		return linspaced;
	}

	double delta = (end - start) / (num - 1);

	for (int i = 0; i < num - 1; ++i)
	{
		linspaced.push_back(start + delta * i);
	}
	linspaced.push_back(end); // I want to ensure that start and end
							  // are exactly the same as the input
	return linspaced;
}
void construct_pp()
{
	for (int j = 0; j < N / 4; j++)
	{
		for (int i = 0; i < N / 2; i++)
		{
			int f = i - 2 * j + 1;
			if (f <= 0)
				pp[i][j] = 0;
			else if (f > 2)
				pp[i][j] = 3;
			else
				pp[i][j] = f;
		}
	}
}
void print_array(int rows = N / 2, int columns = N / 4)
{
	for (int row = 0; row < rows; row++)
	{
		for (int column = 0; column < columns; column++)
			cout << pp[row][column] << " ";
		cout << endl;
	}
}
sm construct_gamma(const int i, const int column = 0)
{
	// recursively construct gamma matrices
	if (column == N / 4 - 1)
		return s[pp[i][N / 4 - 1]];
	else
		return kroneckerProduct(s[pp[i][column]], construct_gamma(i, column + 1));
}
sm left(sm gamma)
{
	return kroneckerProduct(gamma, id);
}
sm left_id(sm gamma)
{
	return left(gamma);
}
sm right(sm gamma)
{
	return kroneckerProduct(g5, gamma);
}
sm right_id(sm gamma)
{
	return kroneckerProduct(id, gamma);
}
bool check_anticommutation()
{
	bool res = false;
	for (int i = 0;i < N / 2; i++)
	{
		MatrixXcd x = (g[i] * g5 + g5 * g[i]).toDense();
		if (x.real().any() || (1i * x).real().any())
			return false;
		for (int j = 0;j < i; j++)
		{
			MatrixXcd x = (g[i] * g[j] + g[j] * g[i]);
			if (x.real().any() || (1i * x).real().any())
				return false;
		}
	}
	return true;
}
void tic()
{
	t0 = high_resolution_clock::now();
}
void toc()
{
	t1 = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(t1 - t0);
	cout << "Finished in " << duration.count() * 1e-3 << "s." << endl;
	t0 = t1;
}
void buildH()
{
	const double gamma = sqrt(6.0 / pow((N / 2.0), 3.0));
	cout << "gamma: " << gamma << endl;
	default_random_engine generator(3);
	normal_distribution<double> distribution(0.0, 1.0);
	ifstream stream("random_numbers.txt", std::ios::in);
	for (int i = 0; i < N / 2; i++)
	{
		for (int j = i + 1; j < N / 2; j++)
		{
			for (int k = j + 1; k < N / 2; k++)
			{
				for (int l = k + 1; l < N / 2; l++)
				{
					//double r = distribution(generator);
					double r;
					stream >> r;
					h += r * g[i] * g[j] * g[k] * g[l];
				}
			}
		}
		Hint += left(g[i]) * right(g[i]);
		cout << i + 1 << " out of " << N / 2 << endl;
	}
	h *= gamma * J / 4.0;
	HLR = (left_id(h) - right_id(h));
	H = (left_id(h) * (1 + eta) + right_id(h) * (1-eta)) + 1i / 2.0 * mu * Hint;
}
void eigs()
{
	//Spectra sparse eigenvalue computation
	SparseMatrix<double> sdr(2, 2);
	SparseMatrix<double> sor(2, 2);
	sdr.setIdentity();
	sor.insert(0, 1) = -1;
	sor.insert(1, 0) = 1;
	SparseMatrix<double> Hr = H.real();
	SparseMatrix<double> Hi = (-1i * H).real();
	SparseMatrix<double> H_real = kroneckerProduct(sdr, Hr) + kroneckerProduct(sor, Hi);

	// Construct matrix operation object using the wrapper class DenseSymMatProd
	SparseSymMatProd<double> op(H_real);
	// Construct eigen solver object, requesting the largest three eigenvalues
	SymEigsSolver< double, SMALLEST_ALGE, SparseSymMatProd<double> > eigs(&op, num_evals, 6*num_evals);
	// Initialize and compute
	eigs.init();
	int nconv = eigs.compute();
	// Retrieve results
	if (eigs.info() != SUCCESSFUL)
		cout << endl << "Error in sparse eigenvalue computation." << endl;
	evalues = eigs.eigenvalues();
	MatrixXd evecs = eigs.eigenvectors();

	VectorXd u = evecs(seq(0, (last + 1) / 2 - 1), last);
	VectorXd v = evecs(seq((last + 1) / 2, last), last);
	gs = u + 1i * v;
	gs = gs / gs.norm();

	//SelfAdjointEigenSolver<MatrixXcd> eigensolver(H, ComputeEigenvectors);
	//ev_syk_test = eigensolver.eigenvalues();
	//cout << ev_syk_test;
}
void overlap()
{
	cout << "Norm of gs: " << gs.norm() << endl;
	overlap_data = MatrixXcd(dimSYK, dimSYK);
	for (int n = 0; n < dimSYK; n++)
	{
		for (int m = 0; m < dimSYK; m++)
		{
			MatrixXcd nn = vecs_syk.col(n);
			VectorXcd mm = vecs_syk.col(m);
			VectorXcd nm = kroneckerProduct(nn, mm);
			//cout << "Norm of nm: " << nm.norm() << endl;
			auto v = nm.dot(gs);
			//cout << v.real() * v.real() + v.imag() * v.imag() << "\t";
			overlap_data(n, m) = v;
		}
		//cout << endl;
	}
}
string d_tostr(double x)
{
	string str = to_string(x);
	replace(str.begin(), str.end(), '.', '_');
	return str;
}
void _save(MatrixXd src, string pathAndName)
{
	ofstream stream(pathAndName, ios::out | ios::trunc);
	if (stream)  // check if open
	{
		stream << "n=" << N << "\n";
		stream << "J=" << J << "\n";
		stream << "eta=" << eta << "\n";
		stream << "mu=" << mu << "\n";
		stream << src.format(CSVFormat) << "\n";
		stream.close();  // close file
	}
	else
	{
		cerr << "Could not open file. Error." << endl;
	}
}

int main(int argc, char** argv)
{
	eta = 0;
	mu = 0.2;
	if (argc > 1) eta = atof(argv[1]);
	if (argc > 2) mu = atof(argv[2]);

	id.setIdentity();
	define_pauli();
	construct_pp();
	//print_array();
	//cout << endl << endl;

	tic();
	cout << "n = " << N << endl;
	cout << "J = " << J << endl;
	cout << "mu = " << mu << endl;
	cout << "eta = " << eta << endl;
	cout << "Beginning to construct gamma matrices... ";
	for (int i = 0;i < N / 2; i++)
	{
		sm gg = construct_gamma(i);
		g.push_back(gg);
		if (i == 0)
			g5 = gg;
		else
			g5 = g5 * gg;
	}
	toc();

	cout << endl << "Do all matrices anticommute? (1=yes): " << check_anticommutation() << endl;
	cout << endl << "Now building Hamiltonian... " << endl;
	buildH();
	auto hSYK_dense = h.toDense();
	toc();

	cout << endl << "Computing eigenvalues of single SYK... ";
	//ComplexEigenSolver<MatrixXcd> eigensolver(hSYK_dense, true);
	SelfAdjointEigenSolver<MatrixXcd> eigensolver(hSYK_dense, ComputeEigenvectors);
	ev_syk = eigensolver.eigenvalues();
	vecs_syk = eigensolver.eigenvectors();
	toc();


	cout << endl << "Computing lowest eigenvalue of coupled SYKs... ";
	eigs();
	toc();
	cout << "Lowest Eigenvalues:" << endl << evalues << endl;

	cout << endl << "Norm of (H_L-H_R)|GS>: ";
	MatrixXcd HLRgs = HLR * gs;
	cout << (HLRgs).norm() << endl;

	overlap();

	_save(overlap_data.real(), "data\\" + to_string(N) + "n" + d_tostr(eta) + "eta" + d_tostr(mu) + "mu_overlap_real.txt");
	_save((-1i * overlap_data).real(), "data\\" + to_string(N) + "n" + d_tostr(eta) + "eta" + d_tostr(mu) + "mu_overlap_imag.txt");
	_save(evalues, "data\\" + to_string(N) + "n" + d_tostr(eta) + "eta" + d_tostr(mu) + "mu_energies.txt");
	//_save(HLRgs.real(), to_string(N) + "n_HLRgs_real.txt");
	//_save((-1i * HLRgs).real(), to_string(N) + "n_HLRgs_imag.txt");
	//_save(ev_syk.real(), to_string(N) + "n_ev_syk.txt");
}
