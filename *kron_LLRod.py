from laplace import Laplace
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

# Função para plotar os resultados de regressão
def plot_regression(X_train, y_train, X_test, f_mu, pred_std, file_name="cubic_example_plot", plot=True):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train.cpu().numpy(), y_train.cpu().numpy(), color='blue', label='Dados de Treinamento')
    plt.plot(X_test, f_mu, color='red', label='Média das Predições')
    plt.fill_between(X_test, f_mu - 2*pred_std, f_mu + 2*pred_std, color='orange', alpha=0.3, label='Intervalo de Confiança (2σ)')
    plt.title(file_name)
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

# Função para obter camadas com parâmetros (pesos e vieses separados)
def get_parameter_layers(model):
    parameter_layers = []
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            parameter_layers.append(layer.weight)
            parameter_layers.append(layer.bias)
    return parameter_layers

parameter_layers = get_parameter_layers(model)
n_total_param_layers = len(parameter_layers)  # Total de camadas de parâmetros (pesos e vieses)

# Função para otimizar precisão a priori
def optimize_prior(la, prior_structure):
    if prior_structure == "scalar":
        init_prior_prec = torch.ones(1, device=la._device)
    elif prior_structure == "diag":
        init_prior_prec = torch.ones(la.n_params, device=la._device)
    elif prior_structure == "layerwise":
        # Número de camadas lógicas que estão sendo otimizadas
        # Para 'last_layer', isso será 1
        # Para 'all_layers', será o número de camadas lógicas
        n_prior_params = len(la.prior_structure_group_indices)
        init_prior_prec = torch.ones(n_prior_params, device=la._device)
    else:
        raise ValueError("prior_structure deve ser 'scalar', 'diag' ou 'layerwise'.")

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
    elif prior_structure == "diag":
        prior_prec = la.prior_precision_diag.cpu().detach().numpy()
        print(f"Vetor de Precisão a Priori ({prior_structure.capitalize()}):")
        print(prior_prec)
    elif prior_structure == "layerwise":
        prior_prec = la.prior_precision_diag.cpu().detach().numpy()
        print(f"Vetor de Precisão a Priori ({prior_structure.capitalize()}):")
        print(prior_prec)

    return prior_prec

# Função para reconstruir a matriz de precisão a priori
def reconstruct_precision_matrix(la, prior_prec, prior_structure, parameter_layers, subset_of_weights):
    # Calcular o total de parâmetros
    total_params = sum(layer.numel() for layer in parameter_layers)
    full_precision = np.eye(total_params)

    if prior_structure == "scalar":
        # Substituir toda a diagonal por um valor escalar
        full_precision = np.full((total_params, total_params), prior_prec)
        np.fill_diagonal(full_precision, prior_prec)
    elif prior_structure == "diag":
        if subset_of_weights == "all_layers":
            # Substituir toda a diagonal por precisões individuais
            full_precision = np.diag(prior_prec)
        elif subset_of_weights == "last_layer":
            # Substituir apenas a última camada
            # Identificar os índices da última camada
            last_layer_weights = parameter_layers[-2]
            last_layer_bias = parameter_layers[-1]
            n_last_layer_weights = last_layer_weights.numel()
            n_last_layer_bias = last_layer_bias.numel()
            total_last_layer_params = n_last_layer_weights + n_last_layer_bias

            # Encontrar o início da última camada
            start = sum(layer.numel() for layer in parameter_layers[:-2])

            # Verificar se prior_prec tem o tamanho correto
            expected_prior_size = total_last_layer_params
            if len(prior_prec) != expected_prior_size:
                raise ValueError(f"Expectativa de tamanho de prior_prec: {expected_prior_size}, mas recebeu: {len(prior_prec)}")

            # Substituir os blocos de pesos e vieses da última camada
            # Pesos da última camada
            full_precision[start:start+n_last_layer_weights, start:start+n_last_layer_weights] = np.diag(prior_prec[:n_last_layer_weights])
            # Bias da última camada
            full_precision[start+n_last_layer_weights:start+total_last_layer_params, start+n_last_layer_weights:start+total_last_layer_params] = np.diag(prior_prec[n_last_layer_weights:])
    elif prior_structure == "layerwise":
        # Substituir blocos de cada camada com valores distintos
        start = 0
        for i, layer in enumerate(parameter_layers):
            n_params_layer = layer.numel()
            # Determinar se esta camada está sendo otimizada
            if subset_of_weights == "all_layers":
                # Todas as camadas estão sendo otimizadas
                prior_value = prior_prec[i // 2]  # Cada par (weight, bias) corresponde a uma camada lógica
            elif subset_of_weights == "last_layer":
                # Apenas a última camada está sendo otimizada
                if i // 2 == (len(parameter_layers) // 2) - 1:  # Última camada lógica
                    prior_value = prior_prec[0]
                else:
                    prior_value = 1.0  # Precisão padrão para outras camadas
            else:
                raise ValueError("subset_of_weights deve ser 'all_layers' ou 'last_layer'.")
            
            full_precision[start:start+n_params_layer, start:start+n_params_layer] = prior_value
            start += n_params_layer
    else:
        raise ValueError("prior_structure deve ser 'scalar', 'diag' ou 'layerwise'.")

    return full_precision

# Função para verificar se a matriz é diagonal
def is_diag_matrix(matrix):
    return np.allclose(matrix, np.diag(np.diag(matrix)))

# Função para visualizar a matriz de precisão a priori
def plot_precision_matrix(precision_matrix, prior_structure):
    plt.figure(figsize=(12, 12))
    sns.heatmap(precision_matrix, annot=False, cmap="viridis")
    plt.title(f"Matriz de Precisão a Priori ({prior_structure.capitalize()})")
    plt.xlabel("Parâmetros")
    plt.ylabel("Parâmetros")
    plt.show()

# Função para realizar previsões e plotar os resultados
def perform_predictions_and_plot(la, X_test, prior_structure, subset_of_weights, save=True):
    f_mu, f_var = la(X_test)
    f_mu = f_mu.squeeze().detach().cpu().numpy()
    f_sigma = f_var.squeeze().sqrt().cpu().numpy()
    pred_std = np.sqrt(f_sigma*2 + la.sigma_noise.item()*2)
    file_name = f"Rod, cubic, W={subset_of_weights.capitalize()}, H=Kron, P={prior_structure}"
    plot_regression(
        X_train,
        y_train,
        X_test.flatten().cpu().numpy(),
        f_mu,
        pred_std,
        file_name=file_name,
        plot=not save
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
def run_prior_optimization(prior_structure, subset_of_weights, hessian_structure):
    print(f"\n--- Otimização da Precisão a Priori com estrutura '{prior_structure}' e subset '{subset_of_weights}' ---")
    la = Laplace(model, "regression", subset_of_weights=subset_of_weights, hessian_structure=hessian_structure)
    la.fit(train_loader)

    prior_prec = optimize_prior(la, prior_structure)
    precision_matrix = reconstruct_precision_matrix(la, prior_prec, prior_structure, parameter_layers, subset_of_weights)

    if prior_structure == "scalar":
        is_uniform = np.allclose(np.diag(precision_matrix), prior_prec)
        print(f"A precisão a priori é uniformemente escalada? {is_uniform}")
    elif prior_structure == "diag":
        if subset_of_weights == "all_layers":
            is_diag = is_diag_matrix(precision_matrix)
        elif subset_of_weights == "last_layer":
            # Verificar se a matriz é diagonal
            is_diag = is_diag_matrix(precision_matrix)
        print(f"A precisão a priori é estritamente diagonal? {is_diag}")
    elif prior_structure == "layerwise":
        # Não é suportado com subset_of_weights='last_layer', conforme o erro
        pass

    plot_precision_matrix(precision_matrix, prior_structure)

    print(f"\n--- Otimização da Variância do Ruído para estrutura '{prior_structure}' ---")
    optimize_sigma_noise(la, prior_structure)
    perform_predictions_and_plot(la, X_test, prior_structure, subset_of_weights, save=True)

    return la

# Função para automatizar todo o processo para diferentes estruturas
def automate_prior_optimization(prior_structures, subset_of_weights, hessian_structure):
    laplace_models = {}
    for structure in prior_structures:
        laplace_models[structure] = run_prior_optimization(structure, subset_of_weights, hessian_structure)
    return laplace_models

# Definição das estruturas de precisão a priori suportadas com base no subset_of_weights
def get_prior_structures(subset_of_weights):
    if subset_of_weights == "last_layer":
        return ["scalar", "diag"]  # Removido 'layerwise' pois não é compatível
    elif subset_of_weights == "all_layers":
        return ["scalar", "diag", "layerwise"]
    else:
        raise ValueError("subset_of_weights deve ser 'all_layers' ou 'last_layer'.")

# Escolha do subset_of_weights: 'all_layers' ou 'last_layer'
subset_of_weights = "last_layer"  # Alterar para "all_layers" conforme necessário

# Obter as estruturas de prior válidas com base no subset_of_weights
prior_structures = get_prior_structures(subset_of_weights)

# Escolha da estrutura do Hessian
hessian_structure = "kron"  # Alterar para "kron", "diag", "lowrank", ou "gp" conforme necessário

# Executar a automatização para as estruturas definidas
laplace_models = automate_prior_optimization(prior_structures, subset_of_weights, hessian_structure)

# Imprime todos os valores da precisão a priori e sigma_noise final
print("\n--- Valores Finais ---")
for structure in prior_structures:
    la = laplace_models[structure]
    if structure == "scalar":
        print(f"Precisão a Priori (Scalar): {la.prior_precision_diag.cpu().detach().numpy()[0]}")
    elif structure == "diag":
        print(f"Vetor de Precisão a Priori (Diag): {la.prior_precision_diag.cpu().detach().numpy()}")
    elif structure == "layerwise":
        print(f"Vetor de Precisão a Priori (Layerwise): {la.prior_precision_diag.cpu().detach().numpy()}")
    print(f"Valor final do sigma_noise para '{structure}': {la.sigma_noise.item():.4f}")