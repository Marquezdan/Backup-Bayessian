from laplace import Laplace
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

# Função para plotar os resultados de regressão
def plot_regression(X_train, y_train, X_test, f_mu, pred_std, prior_value, sigma_value, file_name="cubic_example_plot", plot=True):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train.cpu().numpy(), y_train.cpu().numpy(), color='blue', label='Dados de Treinamento')
    plt.plot(X_test, f_mu, color='red', label='Média das Predições')
    plt.fill_between(X_test, f_mu - 2*pred_std, f_mu + 2*pred_std, color='orange', alpha=0.3, label='Intervalo de Confiança (2σ)')
    plt.title(file_name)  # Usa o valor de file_name como título
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    
    if plot:
        plt.show()
    else:
        plt.savefig(f"{file_name}.png")
        plt.close()


# Configurações e criação dos dados
n_epochs = 1000
torch.manual_seed(711)

def get_cubic_example_with_random_spacing(n, sigma_noise):
    X_train = -3 + 6 * torch.rand(n).reshape(-1, 1)
    y_train = X_train**3 + sigma_noise * torch.randn(X_train.shape)
    X_test = torch.linspace(-4, 4, 100).reshape(-1, 1)
    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=len(X_train), shuffle=True)
    return X_train, y_train, train_loader, X_test

X_train, y_train, train_loader, X_test = get_cubic_example_with_random_spacing(n=20, sigma_noise=0.3)

# Criar e treinar o modelo MAP
def get_model():
    torch.manual_seed(711)
    return torch.nn.Sequential(
        torch.nn.Linear(1, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 1)
    )

model = get_model()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(n_epochs):
    for X, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

# Inicialização da aproximação de Laplace
la = Laplace(model, "regression", subset_of_weights="all", hessian_structure="full")
la.fit(train_loader)

# Função para obter camadas com parâmetros (pesos e vieses separados)
def get_parameter_layers(model):
    parameter_layers = []
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            parameter_layers.append(layer.weight)
            parameter_layers.append(layer.bias)
    return parameter_layers

parameter_layers = get_parameter_layers(model)
n_layers = len(parameter_layers)

# Função para otimizar precisão a priori
def optimize_prior(la, prior_structure):
    if prior_structure == "scalar":
        init_prior_prec = torch.ones(1, device=la._device)
    elif prior_structure == "layerwise":
        init_prior_prec = torch.ones(n_layers, device=la._device)
    elif prior_structure == "diag":
        init_prior_prec = torch.ones(la.n_params, device=la._device)
    else:
        raise ValueError("prior_structure deve ser 'scalar', 'layerwise' ou 'diag'.")

    la.optimize_prior_precision(
        pred_type="nn",
        method="marglik",
        n_steps=100,
        lr=1e-1,
        init_prior_prec=init_prior_prec,
        prior_structure=prior_structure,
        verbose=True,
        progress_bar=True
    )

    if prior_structure == "scalar":
        prior_prec = la.prior_precision_diag.cpu().detach().numpy()[0]
        print("Valor de Precisão a Priori (Scalar):", prior_prec)
    else:
        prior_prec = la.prior_precision_diag.cpu().detach().numpy()
        print(f"Vetor de Precisão a Priori ({prior_structure.capitalize()}):")
        print(prior_prec)

    return prior_prec

# Função corrigida para reconstruir a matriz de precisão a priori
def reconstruct_precision_matrix(la, prior_prec, prior_structure, parameter_layers):
    if prior_structure == "scalar":
        precision_matrix = np.zeros((la.n_params, la.n_params))
        np.fill_diagonal(precision_matrix, prior_prec)
    elif prior_structure == "layerwise":
        precision_matrix = np.zeros((la.n_params, la.n_params))
        start = 0
        for i, layer in enumerate(parameter_layers):
            n_params_layer = layer.numel()
            precision_matrix[start:start+n_params_layer, start:start+n_params_layer] = prior_prec[i]
            start += n_params_layer
    elif prior_structure == "diag":
        precision_matrix = np.diag(prior_prec)
    else:
        raise ValueError("prior_structure deve ser 'scalar', 'layerwise' ou 'diag'.")

    return precision_matrix

# Função para verificar se a matriz é block-diagonal
def is_block_diag_matrix(matrix, parameter_layers):
    start = 0
    for layer in parameter_layers:
        n_params_layer = layer.numel()
        end = start + n_params_layer
        if not np.allclose(matrix[start:end, :start], 0) or not np.allclose(matrix[:start, start:end], 0):
            return False
        start = end
    return True

# Função para visualizar a matriz de precisão a priori
def plot_precision_matrix(precision_matrix, prior_structure):
    plt.figure(figsize=(12, 12))
    sns.heatmap(precision_matrix, annot=False, cmap="viridis")
    plt.title(f"Matriz de Precisão a Priori ({prior_structure.capitalize()})")
    plt.xlabel("Parâmetros")
    plt.ylabel("Parâmetros")
    plt.show()

# Função para realizar previsões e plotar os resultados
def perform_predictions_and_plot(la, X_test, prior_structure, save=True):
    f_mu, f_var = la(X_test)
    f_mu = f_mu.squeeze().detach().cpu().numpy()
    f_sigma = f_var.squeeze().sqrt().cpu().numpy()
    pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item()**2)
    file_name = f"cubic, W=All, H=Full Prior={prior_structure}"

    # Calcular prior_value e sigma_value
    if prior_structure == "scalar":
        prior_value = la.prior_precision_diag.mean().item()
    elif prior_structure == "layerwise":
        prior_value = la.prior_precision_diag.mean().item()
    else:
        prior_value = None

    sigma_value = la.sigma_noise.item()

    # Chamar a função de plotagem
    plot_regression(
        X_train,
        y_train,
        X_test.flatten().cpu().numpy(),
        f_mu,
        pred_std,
        file_name=file_name,
        plot=not save,
        prior_value=prior_value,
        sigma_value=sigma_value
    )

    if save:
        print(f"Plot salvo como '{file_name}.png'.")

# Função para otimizar sigma_noise
def optimize_sigma_noise(la, prior_structure):
    log_sigma = torch.ones(1, requires_grad=True, device=la._device)
    sigma_optimizer = torch.optim.Adam([log_sigma], lr=1e-1)
    sigma_n_epochs = 100

    for epoch in range(sigma_n_epochs):
        sigma_optimizer.zero_grad()
        neg_marglik = - la.log_marginal_likelihood(la.prior_precision_diag, log_sigma.exp())
        neg_marglik.backward()
        sigma_optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{sigma_n_epochs}, Negative Marginal Likelihood: {neg_marglik.item():.4f}, sigma_noise: {log_sigma.exp().item():.4f}")

    la.sigma_noise = log_sigma.exp().detach()
    print(f"Valor final do sigma_noise para '{prior_structure}': {la.sigma_noise.item():.4f}")
    return la.sigma_noise.item()

# Função principal para automatizar a otimização e visualização
def run_prior_optimization(prior_structure):
    print(f"\n--- Otimização da Precisão a Priori com estrutura '{prior_structure}' ---")
    prior_prec = optimize_prior(la, prior_structure)
    precision_matrix = reconstruct_precision_matrix(la, prior_prec, prior_structure, parameter_layers)

    if prior_structure == "scalar":
        is_uniform = np.allclose(np.diag(precision_matrix), prior_prec)
        print(f"A precisão a priori é uniformemente escalada? {is_uniform}")
    elif prior_structure == "layerwise":
        is_block_diag = is_block_diag_matrix(precision_matrix, parameter_layers)
        print(f"A precisão a priori é estritamente block-diagonal por camada? {is_block_diag}")
    elif prior_structure == "diag":
        is_diag = np.allclose(precision_matrix, np.diag(np.diag(precision_matrix)))
        print(f"A precisão a priori é estritamente diagonal? {is_diag}")

    plot_precision_matrix(precision_matrix, prior_structure)

# Função para automatizar todo o processo para diferentes estruturas
def automate_prior_optimization(prior_structures):
    for structure in prior_structures:
        run_prior_optimization(structure)
        print(f"\n--- Otimização da Variância do Ruído para estrutura '{structure}' ---")
        optimize_sigma_noise(la, structure)
        perform_predictions_and_plot(la, X_test, structure, save=True)

# Definição das estruturas de precisão a priori suportadas
prior_structures = ["scalar", "layerwise", "diag"]

# Executar a automatização para as estruturas definidas
automate_prior_optimization(prior_structures)

# Imprime todos os valores da precisão a priori e sigma_noise final
print("\n--- Valores Finais ---")
for structure in prior_structures:
    if structure == "scalar":
        print(f"Precisão a Priori (Scalar): {la.prior_precision_diag.cpu().detach().numpy()[0]}")
    elif structure == "layerwise":
        print(f"Precisão a Priori (Layerwise): {la.prior_precision_diag.cpu().detach().numpy()}")
    elif structure == "diag":
        print(f"Precisão a Priori (Diag): {la.prior_precision_diag.cpu().detach().numpy()}")
print(f"Valor final do sigma_noise: {la.sigma_noise.item():.4f}")