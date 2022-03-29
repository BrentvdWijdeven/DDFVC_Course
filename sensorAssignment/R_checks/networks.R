'%>%' <- dplyr::'%>%'

nodes_df <- read.csv('data/nodes_df.csv')
edges_df <- read.csv('data/edges_df.csv')

?igraph::graph.data.frame
net <- igraph::graph.data.frame(edges_df,
                                vertices=nodes_df,
                                )

net
igraph::vertex.attributes(net)
igraph::edge.attributes(net)

plot(net)

ggraph::ggraph(net, layout = 'nicely') + 
  ggraph::geom_node_point(ggplot2::aes(color=huge_interpolated), size=5) +
  ggraph::geom_node_text(ggplot2::aes(label=name), repel=TRUE) +
  ggraph::geom_edge_link()

ggraph::ggraph(net, layout = 'nicely') + 
  ggraph::geom_node_point(ggplot2::aes(color=autocorr), size=5) +
  ggraph::geom_edge_link()

ggraph::ggraph(net, layout = 'nicely') + 
  ggraph::geom_node_point(ggplot2::aes(color=type), size=5) +
  # ggraph::geom_node_text(ggplot2::aes(label=name), repel=TRUE) +
  ggraph::geom_edge_link()

## PLOT SENSOR NAMES  
# ggraph::geom_node_text(ggplot2::aes(label=name), repel=TRUE)

igraph::E(net)$sign_pos <- igraph::E(net)$sign + 2
ggraph::ggraph(net, layout = 'nicely') + 
  ggraph::geom_node_point(ggplot2::aes(color=type), size=5) +
  # ggraph::geom_node_text(ggplot2::aes(label=name), repel=TRUE) +
  ggraph::geom_edge_link(ggplot2::aes(color=sign_pos))

max_cliques <- igraph::max_cliques(net)

igraph::cluster_walktrap(net) %>% 
  plot(., net)










