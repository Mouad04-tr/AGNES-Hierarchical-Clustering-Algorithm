from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Définition des transactions
transactions = [
    ['Limonade', 'Thé', 'Gâteau', 'Sucre'],
    ['Gâteau', 'Thé', 'Chocolat', 'Miel', 'Sucre'],
    ['Chocolat', 'Thé', 'Sucre', 'Miel'],
    ['Sucre', 'Thé', 'Gâteau']
]

# Encodage des transactions
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Extraction des itemsets fréquents
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

# Affichage des règles d'association
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)
print("Règles d'association :\n", rules)
