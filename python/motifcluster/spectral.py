import numpy as np
from scipy.sparse import linalg
from scipy import sparse

#' Compute first few eigenvalues and eigenvectors of a matrix
#'
#' Compute the first few eigenvalues (by magnitude) and
#' associated eigenvectors of a matrix.
#' @param mat Symmetric matrix for which eigenvalues and eigenvectors
#' are to be calculated.
#' @param num_eigs Number of eigenvalues and eigenvectors to calculate.
#' @return A list with two entries:
#' \code{vals} contains a length-\code{num_eigs} vector of the first few
#' eigenvalues,
#' and vects contains an \code{nrow(mat)} by \code{num_eigs} matrix
#' of the associated eigenvectors.
#' @importFrom RSpectra eigs
#' @keywords internal

def get_first_eigs(mat, num_eigs):

  # check args
  mat = sparse.csr_matrix(mat, dtype = "f")
  assert num_eigs == np.floor(num_eigs)
  assert num_eigs >= 1

  # get spectrum
  vals, vects = linalg.eigsh(mat, k = num_eigs, which = "SM")

  # order eigenvalues and eigenvectors
  ordering = np.argsort(vals.real)
  vals = vals.real[ordering]
  vects = vects.real[:, ordering]

  # return a list
  spectrum = {
    "vects": vects,
    "vals": vals
  }

  return(spectrum)


#' Build a Laplacian matrix
#'
#' Build a Laplacian matrix (combinatorial Laplacian or random-walk Laplacian)
#' from a symmetric (weighted) graph adjacency matrix.
#' @param adj_mat Symmetric adjacency matrix from which to build the Laplacian.
#' @param type_lap Type of Laplacian to build.
#' One of \code{"comb"} (combinatorial) or \code{"rw"} (random-walk).
#' @return The specified Laplacian matrix.
#' @examples
#' adj_mat = matrix(c(1:9), nrow = 3)
#' build_laplacian(adj_mat, "rw")
#' @export

def build_laplacian(adj_mat, type_lap = "rw"):

  # check args
  assert type_lap in ["comb", "rw"]
  adj_mat = sparse.csr_matrix(adj_mat)
  n = adj_mat.shape[0]

  # initialize parameters
  degs_adj_mat = adj_mat.sum(axis = 0).reshape(n)

  # combinatorial Laplacian
  if type_lap == "comb":
    degs_matrix = sparse.diags(degs_adj_mat, offsets = [0], shape = (n, n))
    L =  degs_matrix - adj_mat

  # random-walk Laplacian
  elif type_lap == "rw":
    assert (degs_adj_mat > 0).all()
    inv_degs_matrix = sparse.diags(1 / degs_adj_mat, offsets = [0], shape = (n, n))
    L =  sparse.identity(n) - inv_degs_matrix * adj_mat

  L = sparse.csr_matrix(L)

  return L

#' Run Laplace embedding
#'
#' Run Laplace embedding on a symmetric (weighted) adjacency matrix
#' with a specified number of eigenvalues and eigenvectors.
#' @param adj_mat Symmetric adjacency matrix to be embedded.
#' @param num_eigs Number of eigenvalues and eigenvectors for the embedding.
#' @param type_lap Type of Laplacian for the embedding.
#' One of \code{"comb"} (combinatorial) or \code{"rw"} (random-walk).
#' @return A list with two entries:
#' \code{vals} contains the length-\code{num_eigs} vector
#' of the first few eigenvalues of the Laplacian,
#' and \code{vects} contains an \code{nrow(adj_mat)} by \code{num_eigs} matrix
#' of the associated eigenvectors.
#' @examples
#' adj_mat = matrix(c(1:9), nrow = 3)
#' run_laplace_embedding(adj_mat, 2, "rw")
#' @export

#run_laplace_embedding = function(adj_mat, num_eigs,
                                  #type_lap = c("comb", "rw")) {

  # check args
  #if (!all.equal(num_eigs, as.integer(num_eigs))) {
    #stop("num_eigs must be an integer.")
  #}
  #if (!(num_eigs > 0)) {
    #stop("num_eigs must be at least 1.")
  #}
  #type_lap = match.arg(type_lap)

  # build and embed Laplacian
  #laplacian = build_laplacian(adj_mat, type_lap)
  #spectrum = get_first_eigs(laplacian, num_eigs)

  #return(spectrum)
#}

#' Run motif embedding
#'
#' Calculate a motif adjacency matrix for a given motif and motif type,
#' restrict it to its largest connected component,
#' and then run Laplace embedding with specified Laplacian type and
#' number of eigenvalues and eigenvectors.
#' @param adj_mat Adjacency matrix to be embedded.
#' @param motif_name Motif used for the motif adjacency matrix.
#' @param motif_type Type of motif adjacency matrix to use.
#' One of \code{"func"} or \code{"struc"}.
#' @param mam_weight_type Weighting scheme for the motif adjacency matrix.
#' One of \code{"unweighted"}, \code{"mean"} or \code{"product"}.
#' @param mam_method The method to use for building the motif adjacency matrix.
#' One of \code{"sparse"} or \code{"dense"}.
#' @param num_eigs Number of eigenvalues and eigenvectors for the embedding.
#' @param type_lap Type of Laplacian for the embedding.
#' One of \code{"comb"} or \code{"rw"}.
#' @return A list with 7 entries:
#' \itemize{
#'   \item \code{adj_mat}: the original adjacency matrix.
#'   \item \code{motif_adj_mat}: the motif adjacency matrix.
#'   \item \code{comps}: the indices of the largest connected component
#'     of the motif adjacency matrix.
#'   \item \code{adj_mat_comps}: the original adjacency matrix restricted
#'     to the largest connected component of the motif adjacency matrix.
#'   \item \code{motif_adj_mat_comps}: the motif adjacency matrix restricted
#'     to its largest connected component.
#'   \item \code{vals}: a length-\code{num_eigs} vector containing the
#'     eigenvalues associated with the Laplace embedding
#'     of the restricted motif adjacency matrix.
#'   \item \code{vects}: a matrix
#'     containing the eigenvectors associated with the Laplace embedding
#'     of the restricted motif adjacency matrix.
#' }
#' @examples
#' adj_mat = matrix(c(1:9), nrow = 3)
#' run_motif_embedding(adj_mat, "M1", "func", "mean", "sparse", 2, "rw")
#' @export

#run_motif_embedding = function(adj_mat, motif_name,
                       #motif_type = c("struc", "func"),
                       #mam_weight_type = c("unweighted", "mean", "product"),
                       #mam_method = c("sparse", "dense"),
                       #num_eigs, type_lap = c("comb", "rw")) {

  # check args
  #adj_mat = drop0(adj_mat)
  #if (!(motif_name %in% get_motif_names())) {
    #stop("Invalid motif name.")
  #}
  #motif_type = match.arg(motif_type)
  #if (!all.equal(num_eigs, as.integer(num_eigs))) {
    #stop("num_eigs must be an integer.")
  #}
  #mam_weight_type = match.arg(mam_weight_type)
  #mam_method = match.arg(mam_method)
  #if (!(num_eigs > 0)) {
    #stop("num_eigs must be at least 1.")
  #}
  #type_lap = match.arg(type_lap)

  # build motif adjacency matrix
  #motif_adj_mat = build_motif_adjacency_matrix(adj_mat, motif_name,
                     #motif_type, mam_weight_type, mam_method)

  # restrict to largest connected component
  #comps = get_largest_component(motif_adj_mat)
  #adj_mat_comps = adj_mat[comps, comps, drop = FALSE]
  #motif_adj_mat_comps = motif_adj_mat[comps, comps, drop = FALSE]

  # Laplace embedding
  #spect = run_laplace_embedding(motif_adj_mat_comps, num_eigs, type_lap)

  # return list
  #spectrum = list()
  #spectrum$adj_mat = adj_mat
  #spectrum$motif_adj_mat = motif_adj_mat
  #spectrum$comps = comps
  #spectrum$adj_mat_comps = adj_mat_comps
  #spectrum$motif_adj_mat_comps = motif_adj_mat_comps
  #spectrum$vals = spect$vals
  #spectrum$vects = spect$vects

  #return(spectrum)
#}
